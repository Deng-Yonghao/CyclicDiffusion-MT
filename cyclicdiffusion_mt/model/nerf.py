"""Differentiable NeRF: internal <-> Cartesian coordinate conversion."""
import torch, torch.nn as nn

def normalize(v, dim=-1, eps=1e-8):
    return v / (torch.norm(v, dim=dim, keepdim=True) + eps)

def rotate_around_axis(v, axis, angle):
    """Rodrigues rotation. v,axis:(B,3) angle:(B,) -> (B,3)"""
    axis = normalize(axis)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    cross_av = torch.cross(axis, v, dim=-1)
    dot_av = (axis * v).sum(dim=-1, keepdim=True)
    return v * cos_a.unsqueeze(-1) + cross_av * sin_a.unsqueeze(-1) + axis * dot_av * (1 - cos_a.unsqueeze(-1))

def place_atom(r_im2, r_im1, r_i, b, alpha, tau):
    """Place atom given 3 ref atoms. b,alpha,tau:(B,) -> (B,3)"""
    u = normalize(r_i - r_im1)
    v = normalize(r_im1 - r_im2)
    n = normalize(torch.cross(u, v, dim=-1))
    u_rot = rotate_around_axis(u, n, torch.pi - alpha)
    u_final = rotate_around_axis(u_rot, u, tau)
    return r_i + b.unsqueeze(-1) * u_final

class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_frame = nn.Parameter(torch.randn(3, 3) * 0.1)

    def forward(self, bonds, angles, torsions, aa_types):
        """bonds/angles:(B,L,14) torsions:(B,L,7) aa_types:(B,L) -> coords:(B,L,14,3)"""
        B, L = bonds.shape[0], bonds.shape[1]
        device = bonds.device
        coords = torch.zeros(B, L, 14, 3, device=device)
        for b_idx in range(B):
            for i in range(L):
                aa = aa_types[b_idx, i].item()
                n_atoms = self._num_atoms(aa)
                for a in range(n_atoms):
                    if i == 0 and a < 3:
                        coords[b_idx, i, a] = self.init_frame[a]
                    else:
                        r_im2, r_im1, r_i = self._ref_atoms(coords, b_idx, i, a)
                        bond = bonds[b_idx, i, a]
                        ang = angles[b_idx, i, a]
                        tau = self._get_torsion(torsions, b_idx, i, a)
                        coords[b_idx, i, a] = place_atom(
                            r_im2.unsqueeze(0), r_im1.unsqueeze(0),
                            r_i.unsqueeze(0), bond.unsqueeze(0),
                            ang.unsqueeze(0), tau.unsqueeze(0)
                        ).squeeze(0)
        return coords

    def inverse(self, coords, aa_types):
        """coords:(B,L,14,3) -> bonds:(B,L,14), angles:(B,L,14), torsions:(B,L,7)"""
        B, L = coords.shape[0], coords.shape[1]
        device = coords.device
        bonds = torch.zeros(B, L, 14, device=device)
        angles = torch.zeros(B, L, 14, device=device)
        torsions = torch.zeros(B, L, 7, device=device)
        for b_idx in range(B):
            for i in range(L):
                aa = aa_types[b_idx, i].item()
                n = self._num_atoms(aa)
                for a in range(1, n):
                    bonds[b_idx,i,a] = torch.norm(coords[b_idx,i,a] - coords[b_idx,i,a-1])
                    if a > 1:
                        v1 = coords[b_idx,i,a-2] - coords[b_idx,i,a-1]
                        v2 = coords[b_idx,i,a] - coords[b_idx,i,a-1]
                        c = ((v1*v2).sum()/(torch.norm(v1)*torch.norm(v2)+1e-8)).clamp(-1,1)
                        angles[b_idx,i,a] = torch.acos(c)
                    if a > 2:
                        tau = self._dihedral(coords[b_idx,i,a-3],coords[b_idx,i,a-2],coords[b_idx,i,a-1],coords[b_idx,i,a])
                        if a < 4: torsions[b_idx,i,min(a,2)] = tau
                        elif a-4 < 4: torsions[b_idx,i,3+a-4] = tau
        return bonds, angles, torsions

    def _num_atoms(self, aa):
        from cyclicdiffusion_mt.utils.constants import AA_ATOM_COUNT, IDX_TO_AA
        return AA_ATOM_COUNT.get(IDX_TO_AA.get(aa,'ALA'),5)

    def _ref_atoms(self, coords, b, i, a):
        if a < 3: return self.init_frame[0], self.init_frame[1], self.init_frame[2]
        return coords[b,i,a-3], coords[b,i,a-2], coords[b,i,a-1]

    def _get_torsion(self, torsions, b, i, a):
        if a < 4: return torsions[b,i,min(a,2)]
        chi = a-4; return torsions[b,i,3+chi] if chi < 4 else torch.tensor(0.0,device=torsions.device)

    def _dihedral(self, r1,r2,r3,r4):
        b1,b2,b3 = r2-r1, r3-r2, r4-r3
        n1 = normalize(torch.cross(b1.unsqueeze(0),b2.unsqueeze(0)).squeeze(0))
        n2 = normalize(torch.cross(b2.unsqueeze(0),b3.unsqueeze(0)).squeeze(0))
        m1 = torch.cross(n1.unsqueeze(0), normalize(b2.unsqueeze(0)).squeeze(0)).squeeze(0)
        x = (n1*n2).sum()
        y = (m1*n2).sum()
        return torch.atan2(y, x)
