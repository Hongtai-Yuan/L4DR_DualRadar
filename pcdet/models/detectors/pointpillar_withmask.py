from .detector3d_template import Detector3DTemplate
from ...datasets.processor.data_processor import VoxelGeneratorWrapper
import numpy as np
import torch
import queue
from ...utils import common_utils, commu_utils

def _safe_batch_size(coords, fallback_points=None, fallback_bs=None):
    """
    安全获取 batch_size：
    1) 优先用 coords[:,0].max()+1（非空）
    2) 其次用 fallback_bs（若 batch_dict 里有）
    3) 再用 fallback_points[:,0].max()+1（非空）
    4) 最后回退 1
    """
    if isinstance(coords, torch.Tensor) and coords.numel() > 0:
        return int(coords[:, 0].max().item()) + 1
    if fallback_bs is not None:
        return int(fallback_bs)
    if isinstance(fallback_points, torch.Tensor) and fallback_points.numel() > 0:
        return int(fallback_points[:, 0].max().item()) + 1
    return 1


class PointPillarMask(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.TP = common_utils.AverageMeter()
        self.P = common_utils.AverageMeter()
        self.TP_FN = common_utils.AverageMeter()
        self.NOR = common_utils.AverageMeter()
        self.TP_FP_FN = common_utils.AverageMeter()
        self.All = common_utils.AverageMeter()

    def forward(self, batch_dict):
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.point_head(batch_dict)
        pre_mask = batch_dict['point_cls_scores'] > 0.2
        labels = batch_dict['point_cls_labels'] > 0
        self.TP.update((pre_mask&labels).sum().item())
        self.P.update(pre_mask.sum().item())
        self.TP_FP_FN.update((pre_mask|labels).sum().item())
        self.NOR.update((~(pre_mask^labels)).sum().item())
        self.All.update(pre_mask.shape[0])
        self.TP_FN.update(labels.sum().item())
        # if self.TP.avg>0:
        #     print("recall = ",round(self.TP.sum/self.TP_FN.sum,4),end=" ; ")
        #     print("precise = ",round(self.TP.sum/self.P.sum,4),end=" ; ")
        #     print("mIoU = ",round(self.TP.sum/self.TP_FP_FN.sum,4),end=" ; ")
        #     print("PA = ",round(self.NOR.sum/self.All.sum,4))
        batch_dict['raw_radar_points'] = batch_dict['radar_points']
        batch_dict['radar_points'] = torch.cat([batch_dict['radar_points'][pre_mask],batch_dict['point_cls_scores'][pre_mask].reshape(-1,1)], dim=1)
        batch_dict = self.transform_points_to_voxels(batch_dict, batch_dict['radar_points'])
        batch_dict = self.vfe(batch_dict)
        
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if 'cen_loss' in batch_dict:
                cen_loss = batch_dict['cen_loss']
                loss = loss + cen_loss
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            pred_dicts[0]['batch_dict'] = batch_dict
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        
        loss = loss_rpn
        if self.model_cfg.get('POINT_HEAD', None) is not None:
            loss_point, tb_dict = self.point_head.get_loss()
            loss = loss_rpn + loss_point
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        
        
        return loss, tb_dict, disp_dict

    def transform_points_to_voxels(self, batch_dict, bs_radar_points):
        MAX_NUMBER_OF_VOXELS = {
            'train': 16000,
            'test': 40000
        }
        self.voxel_generator_r = VoxelGeneratorWrapper(
            vsize_xyz=[0.16, 0.16, 4],
            coors_range_xyz=self.dataset.data_processor.point_cloud_range,
            num_point_features=self.dataset.data_processor.num_point_features[1] + 1,
            max_num_points_per_voxel=32,
            max_num_voxels=MAX_NUMBER_OF_VOXELS[self.dataset.data_processor.mode],
        )

        # 设备安全：尽量跟随现有张量
        device = None
        if isinstance(batch_dict.get('lidar_voxel_coords', None), torch.Tensor):
            device = batch_dict['lidar_voxel_coords'].device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 如果这一批雷达点为空，直接写入空体素并返回
        if not isinstance(bs_radar_points, torch.Tensor) or bs_radar_points.numel() == 0:
            batch_dict['radar_voxel_coords'] = torch.empty((0, 4), dtype=torch.int32, device=device)
            batch_dict['radar_voxels'] = torch.empty((0, 32, self.dataset.data_processor.num_point_features[1] + 1),
                                                     dtype=torch.float32, device=device)
            batch_dict['radar_voxel_num_points'] = torch.empty((0,), dtype=torch.int32, device=device)
            return batch_dict

        # to numpy
        bs_radar_points_np = bs_radar_points.detach().cpu().numpy()

        # ✅ 关键修改：安全获取 batch_size（替换你原来的 max() 行）
        batch_size = _safe_batch_size(
            batch_dict.get('lidar_voxel_coords', None),
            fallback_points=batch_dict.get('radar_points', None),
            fallback_bs=batch_dict.get('batch_size', None)
        )

        radar_voxels_list = []
        radar_coordinates_list = []
        radar_num_points_list = []

        for bs in range(batch_size):
            bs_mask = bs_radar_points_np[:, 0] == bs
            # 这一张里可能没有雷达点：写入空占位，继续
            if not np.any(bs_mask):
                radar_voxels = np.zeros((0, 32, self.dataset.data_processor.num_point_features[1] + 1),
                                        dtype=np.float32)
                radar_coordinates = np.zeros((0, 3), dtype=np.int32)
                radar_num_points = np.zeros((0,), dtype=np.int32)
            else:
                radar_points = bs_radar_points_np[bs_mask, 1:]  # 去掉 batch 索引列
                voxel_out = self.voxel_generator_r.generate(radar_points)
                radar_voxels, radar_coordinates, radar_num_points = voxel_out
                if not batch_dict['use_lead_xyz'][0]:
                    # remove xyz
                    radar_voxels = radar_voxels[..., 3:]

            radar_voxels_list.append(radar_voxels)
            radar_coordinates_list.append(radar_coordinates)
            radar_num_points_list.append(radar_num_points)

        # 拼 batch 维，并在坐标前面 pad batch 索引
        coors = []
        for i, coor in enumerate(radar_coordinates_list):
            if coor.size == 0:
                coor_pad = np.zeros((0, 4), dtype=np.int32)
            else:
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i).astype(np.int32)
            coors.append(coor_pad)

        # 可能全部为空，也要返回 shape 正确的空张量
        if len(coors) == 0 or sum([c.shape[0] for c in coors]) == 0:
            batch_dict['radar_voxel_coords'] = torch.empty((0, 4), dtype=torch.int32, device=device)
            batch_dict['radar_voxels'] = torch.empty((0, 32, self.dataset.data_processor.num_point_features[1] + 1),
                                                     dtype=torch.float32, device=device)
            batch_dict['radar_voxel_num_points'] = torch.empty((0,), dtype=torch.int32, device=device)
            return batch_dict

        batch_dict['radar_voxel_coords'] = torch.tensor(np.concatenate(coors, axis=0), dtype=torch.int32, device=device)

        # 注意：如果前面去掉了 xyz，这里的通道数已变（不再是 num_point_features[1] + 1），但我们只是透传给后续 VFE；
        # 这里保持与上文一致，不强行检查通道。
        batch_dict['radar_voxels'] = torch.tensor(np.concatenate(radar_voxels_list, axis=0), dtype=torch.float32,
                                                  device=device)
        batch_dict['radar_voxel_num_points'] = torch.tensor(np.concatenate(radar_num_points_list, axis=0),
                                                            dtype=torch.int32, device=device)
        return batch_dict
