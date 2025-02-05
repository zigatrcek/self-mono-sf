from . import kitti_2015_train
from . import kitti_2015_test

from . import kitti_raw_monosf
from . import kitti_raw_monodepth

from . import kitti_comb_mnsf
from . import kitti_eigen_test

from . import modd2_raw_monosf
from . import mastr1325_seg
from . import mods

KITTI_2015_Train_Full_mnsf = kitti_2015_train.KITTI_2015_MonoSceneFlow_Full
KITTI_2015_Train_Full_monodepth = kitti_2015_train.KITTI_2015_MonoDepth_Full

KITTI_2015_Test = kitti_2015_test.KITTI_2015_Test

KITTI_Raw_KittiSplit_Train_mnsf = kitti_raw_monosf.KITTI_Raw_KittiSplit_Train
KITTI_Raw_KittiSplit_Valid_mnsf = kitti_raw_monosf.KITTI_Raw_KittiSplit_Valid
KITTI_Raw_KittiSplit_Full_mnsf = kitti_raw_monosf.KITTI_Raw_KittiSplit_Full
KITTI_Raw_EigenSplit_Train_mnsf = kitti_raw_monosf.KITTI_Raw_EigenSplit_Train
KITTI_Raw_EigenSplit_Valid_mnsf = kitti_raw_monosf.KITTI_Raw_EigenSplit_Valid
KITTI_Raw_EigenSplit_Full_mnsf = kitti_raw_monosf.KITTI_Raw_EigenSplit_Full

KITTI_Raw_KittiSplit_Train_monodepth = kitti_raw_monodepth.KITTI_Raw_KittiSplit_Train
KITTI_Raw_KittiSplit_Valid_monodepth = kitti_raw_monodepth.KITTI_Raw_KittiSplit_Valid

KITTI_Comb_Train = kitti_comb_mnsf.KITTI_Comb_Train
KITTI_Comb_Val = kitti_comb_mnsf.KITTI_Comb_Val
KITTI_Comb_Full = kitti_comb_mnsf.KITTI_Comb_Full

KITTI_Eigen_Test = kitti_eigen_test.KITTI_Eigen_Test

MODD2_Train_mnsf = modd2_raw_monosf.MODD2_Train
MODD2_Valid_mnsf = modd2_raw_monosf.MODD2_Valid
MODD2_Visualisation_mnsf = modd2_raw_monosf.MODD2_Visualisation
MODD2_Inference_mnsf = modd2_raw_monosf.MODD2_Inference

MaSTr1325_Full = mastr1325_seg.MaSTr1325_Full
Mods_Full = mods.Mods_Full
Mods_Train = mods.Mods_Train
Mods_Valid = mods.Mods_Valid
Mods_Test = mods.Mods_Test
