import pytest, torch
from cyclicdiffusion_mt.model.nerf import NeRF, rotate_around_axis, place_atom

class TestRotateAroundAxis:
    def test_no_rotation(self):
        v = torch.tensor([[1.,0.,0.]]); axis = torch.tensor([[0.,0.,1.]]); angle = torch.tensor([0.])
        assert torch.allclose(rotate_around_axis(v,axis,angle), v, atol=1e-6)
    def test_90_degree(self):
        v = torch.tensor([[1.,0.,0.]]); axis = torch.tensor([[0.,0.,1.]]); angle = torch.tensor([torch.pi/2])
        assert torch.allclose(rotate_around_axis(v,axis,angle), torch.tensor([[0.,1.,0.]]), atol=1e-6)
    def test_preserves_length(self):
        v = torch.randn(5,3); axis = torch.randn(5,3); angle = torch.rand(5)*2*torch.pi
        r = rotate_around_axis(v,axis,angle)
        assert torch.allclose(torch.norm(r,dim=-1), torch.norm(v,dim=-1), atol=1e-5)

class TestPlaceAtom:
    def test_straight_line(self):
        r_im2=torch.tensor([[0.,0.,0.]]); r_im1=torch.tensor([[1.,0.,0.]]); r_i=torch.tensor([[2.,0.,0.]])
        b=torch.tensor([1.5]); alpha=torch.tensor([torch.pi]); tau=torch.tensor([0.])
        r = place_atom(r_im2,r_im1,r_i,b,alpha,tau)
        assert r.shape == (1,3)
        assert torch.allclose(r[:,0], torch.tensor([3.5]), atol=1e-4)

class TestNeRF:
    @pytest.fixture
    def nerf(self): return NeRF()
    def test_init_frame_shape(self, nerf):
        assert nerf.init_frame.shape == (3,3)
    def test_forward_shape(self, nerf):
        B,L=2,5; aa=torch.zeros(B,L,dtype=torch.long)
        bonds=torch.randn(B,L,14); angles=torch.randn(B,L,14); torsions=torch.randn(B,L,7)
        coords = nerf(bonds,angles,torsions,aa)
        assert coords.shape == (B,L,14,3)
    def test_differentiable(self, nerf):
        B,L=1,3; aa=torch.zeros(B,L,dtype=torch.long)
        bonds=torch.randn(B,L,14,requires_grad=True)
        angles=torch.randn(B,L,14,requires_grad=True)
        torsions=torch.randn(B,L,7,requires_grad=True)
        coords=nerf(bonds,angles,torsions,aa); coords.sum().backward()
        assert bonds.grad is not None and torsions.grad is not None
    def test_no_nan(self, nerf):
        B,L=1,6; aa=torch.randint(0,20,(B,L))
        bonds=torch.rand(B,L,14)*0.1+1.3; angles=torch.rand(B,L,14)*0.3+1.9
        torsions=torch.rand(B,L,7)*2*torch.pi-torch.pi
        coords=nerf(bonds,angles,torsions,aa)
        assert not torch.isnan(coords).any()
