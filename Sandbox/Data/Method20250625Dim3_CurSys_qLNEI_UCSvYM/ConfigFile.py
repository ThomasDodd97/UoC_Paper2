import os
from pathlib import Path
import numpy as np

from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel
from ax.api.configs import RangeParameterConfig

class OptimisationSetup_class(object):
    def __init__(self):
        # Set the name of the experiment.
        self.name = "DMIBasedResinDualObjectiveRun"

        # Set the tailored experiment:
        self.TailoredExperiment_str = "Method20250625Dim3"

        self.StockUPR1_UP1vsDeI_Constant_DecPct_flt = 0.709         # 70% UP1 (30% DmI)             # j1
        self.StockI1_CSvsBP_Constant_DecPct_flt = 0.8               # 80% CS (20% BP)               # j2
        self.TargetMassOfIUPR1a11_Constant_g_flt = 2.5              # Target Mass of IUPR           # j3
        self.UPR1_StockUPR1vsDeI_Bounds_DecPct_lis = [0.45,1.00]    # 45-100% StockUPR1 (55-0% DmI) # x1
        self.I2_I1vsCS_Bounds_DecPct_lis = [0.40,1.00]              # 40-100% I1 (60-0% CS)         # x2
        self.IUPR1_UPR1vsI2_Bounds_DecPct_lis = [0.50,0.95]         # 50-95% UPR1 (50-5% I2)        # x3

        self.CuringRegime_lis = [
            {"time_mins_flt":180,"temperature_oc_flt":80},
            {"time_mins_flt":60,"temperature_oc_flt":120}
        ]

        # Set the parameters of the parameter space:
        self.Parameters_lis = [
            RangeParameterConfig(name="x1", parameter_type="float", bounds=tuple(self.UPR1_StockUPR1vsDeI_Bounds_DecPct_lis)),
            RangeParameterConfig(name="x2", parameter_type="float", bounds=tuple(self.I2_I1vsCS_Bounds_DecPct_lis)),
            RangeParameterConfig(name="x3", parameter_type="float", bounds=tuple(self.IUPR1_UPR1vsI2_Bounds_DecPct_lis)),
        ]

        # Set the objective properties dictionary:
        # self.Objectives_dict = {"t1":ObjectiveProperties(minimize=False)}
        self.MetricOne_str = "ucs"
        self.MetricTwo_str = "ym"
        self.MetricOneDescription_str = "Ultimate Compressive Strength (N/mm^-2)"
        self.MetricTwoDescription_str = "Young's Modulus (N/mm^-2)"
        self.Metrics_lis = [self.MetricOne_str,self.MetricTwo_str]
        self.Objectives_str = f"{self.MetricOne_str},{self.MetricTwo_str}"
        self.MetricOneThreshold = str(118)
        self.MetricTwoThreshold = str(2900)
        self.OutcomeConstraints_lis=[f"{self.MetricOne_str} >= {self.MetricOneThreshold}",f"{self.MetricTwo_str} >= {self.MetricTwoThreshold}"]

        # Set the objective retrieval methods vital parameters:
        self.DownsamplingFactor_int = int(5)
        self.HorizonValue_int = int(5)

        # Set the number of trials to be used to build a prior over the parameter space.
        self.NoOfPriorSamples_int = int(56)
        # Deterministic sampling method (grid,pseudorandom,quasirandom)
        self.DeterministicSamplingMethod_str = "manual"
        self.x_mat = np.array([[0.454139,0.726232,0.921546,0.643396,0.672259,0.944245,0.857977,0.580206,0.7809429252636317,0.9899644803657563,0.6993114489688069,0.5219518693652795,0.6114324308833363,0.9762702092136775,0.47410178498232014,0.4937982645354389,0.5477940103178361,0.7532101637527698,0.8158861742505665,0.7902773290823787,0.8351488118754502,0.8815959257625985,0.4644579660011302,0.9644728762871096,0.46014408471695645,0.49420496307519024,0.47436834119135096,0.45,0.45492934898337317,0.4756684864868129,0.4517074612481168,0.4504632868626293,0.5154879324202373,0.5393063856243109,0.5533032493141481,0.49386811042334505,0.5118058273889332,0.5235139042512074,0.5203698972425269,0.4918738080652055,0.944353543301126,0.8841634679757444,1.0,1.0,0.8839686833016127,0.9313642738296193,0.9681771716669141,0.9744133925524976,0.9600547575203904,0.7875028833953088,0.9868735998811942,0.9777063696176134,0.9783871932018853,0.9550087890062249,0.859831130713675,0.9559928141448799],[0.81511,0.425882,0.903889,0.655267,0.950053,0.589033,0.731234,0.529532,0.4721177118656732,0.9971696540516293,0.7713645270752942,0.856256466100261,0.9750588717694351,0.5046479350199141,0.45222924412153026,0.40136436709666856,0.7525584803414318,0.6307979191760675,0.8761356893675063,0.4108755211044075,0.927207579452773,0.5687278527354136,0.6802241469919922,0.44210226496222244,0.8177419986554079,0.7239756461225757,0.9480605405046685,0.8103091560793472,0.6387690898514315,0.7733354221776824,0.8262246078382403,0.819136786328901,0.796162895492875,0.7003913325254715,0.7450182321393721,0.808871844826667,0.804483195802473,0.7952796945729674,0.6199553306935707,0.9067794223607275,0.722758658445932,0.6004204014121486,0.4,0.5638419807213082,0.6001468981655995,0.5826175059314345,0.5537854379711873,0.5831771072504501,0.5869091653122689,0.6181832112582601,0.5963524400932824,0.6012367276999249,0.5762895209097056,0.5895448226493264,0.6222287180699405,0.6011416144344134],[0.544035,0.903916,0.652035,0.800093,0.866246,0.578442,0.779786,0.675667,0.727451567993472,0.9494039181353263,0.5036801249576881,0.8328096192346547,0.6099911874291586,0.9347517859972094,0.5235818690490716,0.8803561321907342,0.6977338704836032,0.6259141768877668,0.7508210736103096,0.5102825798623749,0.9202592674531496,0.8153016587481997,0.9432378075777109,0.7168132832690721,0.7936906704029704,0.5590876294614177,0.5,0.5850334777142253,0.5369779505946625,0.539359374499954,0.5244596800562421,0.554221549291614,0.544462605103526,0.5993982933207371,0.505756022528287,0.5385672167671118,0.5939553766790181,0.5984998971525713,0.6250863598045988,0.5869531723036278,0.5626607555198848,0.7370290229966121,0.5,0.7732426819833264,0.57944908669539,0.6574239370562868,0.5736809949356441,0.5903875349775184,0.5615445768411191,0.758036780650088,0.5483724990873869,0.5671446056169236,0.565319708390328,0.5312718844844598,0.7477125367796229,0.5644221314215654]])

        # Set the sequential optimisation routine parameters:
        self.SequentialTechnique = "Ax"
        # Set the surrogate:
        self.Surrogate = SingleTaskGP
        # Set the Kernel:
        self.Kernel = MaternKernel
        # Set the Kernel Options:
        self.KernelOptions = {"nu": 2.5}
        # Set the acquisition function:
        self.AcquisitionFunction = qLogNoisyExpectedHypervolumeImprovement


        # Set the number of trials to be made at each sequential iteration.
        self.NoOfTrialsPerIteration_int = int(1)
        # Set the number of sequential iterations to be carried out in this experiment.
        self.NoOfIterations_int = int(20)
        # The maximum number of trials for this experiment.
        self.MaxNoOfTrials_int = int(self.NoOfPriorSamples_int + (self.NoOfIterations_int * self.NoOfTrialsPerIteration_int))

        # The path of this file.
        self.InputFilePath_str = str(Path(os.path.abspath(__file__)))
        # The home directory for this experiment.
        self.HomeDirectoryPath_str = str(Path(os.path.abspath(__file__)).parent.absolute())
        # Set the path of the experiments Ax json file.
        self.ExperimentFilePath_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/AxExperiment.json"
        # Set the path of the parameter backup file.
        self.BackupPath_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/Backup.csv"
        # Set the path of the raw mechanical test data directory
        self.RawDataMT_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/RawDataMT"
        # Set the path of the raw mechanical test data dimensions file.
        self.RawDataMTDims_str = str(Path(os.path.abspath(__file__)).parent.absolute()) + "/Dimensions.csv"