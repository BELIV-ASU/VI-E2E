 # calculate the similarity of the two queries
            v = agent_query.squeeze(0)        # [900, 256]
            i = agent_query_i.squeeze(1)      # [900, 256]
            v_norm = F.normalize(v, dim=-1)   # [900, 256]
            i_norm = F.normalize(i, dim=-1)   # [900, 256]
            sim = v_norm @ i_norm.T
            sim_thresh = 0.3
            best_sim, best_idx = sim.max(dim=1)      # best similarity and index for each vehicle
            keep_mask = best_sim > sim_thresh
            veh_idx = torch.where(keep_mask)[0]      # vehicle indices to keep
            infra_idx = best_idx[keep_mask]          # matched infra indices
            matches = list(zip(veh_idx.tolist(), infra_idx.tolist()))
            
            #fuse infra-vehicle queries
            agent_query_fusion = self.vi_agent_fuser(
                query=agent_query[:, veh_idx, :].permute(1, 0, 2),
                key=agent_query_i[infra_idx, :, :].permute(1,0,2), # TODO: [A_i,B,D] -> [M_i,B,D] M = A * fut_mode
                value=agent_query_i[infra_idx, :, :].permute(1,0,2),
                query_pos= None, #veh_agent_pos_embed.permute(1, 0, 2),
                key_pos=None, #infra_agent_pos_embed.permute(1, 0, 2),
                key_padding_mask=agent_mask_i[infra_idx, :].permute(1,0).bool()
            ) #[A_v,B,D]
            
            _, Nv, D = agent_query.shape
            Ni = agent_query_i.shape[0]

            all_v = torch.arange(Nv, device=agent_query.device)
            all_i = torch.arange(Ni, device=agent_query.device)
            unmatched_v = all_v[~keep_mask]                # [900-K]

            # infra unmatched → remove infra_idx
            mask_i = torch.ones(Ni, dtype=torch.bool, device=agent_query.device)
            mask_i[infra_idx] = False
            unmatched_i = all_i[mask_i]
            v_unmatched = agent_query[:, unmatched_v].permute(1,0,2)                   # [900-K,1,256]
            i_unmatched = agent_query_i[unmatched_i]   # [900-K,1,256]
            agent_query = torch.cat([agent_query_fusion,v_unmatched,i_unmatched], dim=0)  # → [1800,1,256]
            target_len = 1800
            cur_len = agent_query.shape[0]

            if cur_len < target_len:
                pad = torch.zeros(target_len - cur_len, 1, D, device=agent_query.device)
                agent_query = torch.cat([agent_query, pad], dim=0)








reference = torch.cat([tmp[...,:2], tmp[...,4:5]], dim=2)
        bs = bev_embed.shape[1]
        
        query_pos = agent_pos
        query = agent_query
        reference_points = reference
        reference_points = reference_points.sigmoid().clone()
        init_reference = reference_points
        query = query.permute(1,0,2)
        query_pos = query_pos.permute(1,0,2)
        
        hs, inter_references = self.agent_fusion_decoder(
            query=query,
            key = None,
            value = bev_embed,
            query_pos = query_pos,
            reference_points = reference_points,
            reg_branches=self.reg_branches_fuse,
            cls_branches=self.cls_branches_fuse,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas)
        
        hs = hs.permute(0,2,1,3)
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl-1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches_fuse[lvl](hs[lvl])
            tmp = self.reg_branches_fuse[lvl](hs[lvl])
            
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid().clone()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid().clone()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
