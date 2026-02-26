import os
from pathlib import Path
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from ax.api.configs import RangeParameterConfig

class OptimisationSetup_class(object):
    def __init__(self):
        # Set the name of the experiment.
        self.name = "TrialIn3D"

        # Set the tailored experiment:
        self.TailoredExperiment_str = "Method20250310Dim3"
        self.ResinDesignationUsed = "12A-DMI"

        self.StockUPR1_UP1vsDeI_Constant_DecPct_flt = 0.709         # 70% UP1 (30% DmI)             # j1
        self.StockI1_CSvsBP_Constant_DecPct_flt = 0.8               # 80% CS (20% BP)               # j2
        self.TargetMassOfIUPR1a11_Constant_g_flt = 2.5              # Target Mass of IUPR           # j3
        self.UPR1_StockUPR1vsDeI_Bounds_DecPct_lis = [0.45,1.00]    # 45-100% StockUPR1 (55-0% DeI) # x1
        self.I2_I1vsCS_Bounds_DecPct_lis = [0.40,1.00]              # 40-100% I1 (90-0% CS)         # x2
        self.IUPR1_UPR1vsI2_Bounds_DecPct_lis = [0.50,0.95]          # 50-95% UPR1 (50-5% I2)      # x3

        self.CuringRegime_lis = [
            {"time_mins_flt":180,"temperature_oc_flt":80},
            {"time_mins_flt":60,"temperature_oc_flt":120}
        ]
        # Set the parameters of the parameter space:
        # self.Parameters_lis = [
        #     {"name":"x1", "type":"range","bounds":self.UPR1_StockUPR1vsDeI_Bounds_DecPct_lis,"value_type":"float"},
        #     {"name":"x2", "type":"range","bounds":self.I2_I1vsCS_Bounds_DecPct_lis,"value_type":"float"},
        #     {"name":"x3", "type":"range","bounds":self.IUPR1_UPR1vsI2_Bounds_DecPct_lis,"value_type":"float"}
        # ]
        self.Parameters_lis = [
            RangeParameterConfig(name="x1", parameter_type="float", bounds=tuple(self.UPR1_StockUPR1vsDeI_Bounds_DecPct_lis)),
            RangeParameterConfig(name="x2", parameter_type="float", bounds=tuple(self.I2_I1vsCS_Bounds_DecPct_lis)),
            RangeParameterConfig(name="x3", parameter_type="float", bounds=tuple(self.IUPR1_UPR1vsI2_Bounds_DecPct_lis)),
        ]

        self.MetricOne_str = "t1"
        self.MetricOneDescription_str = "Ultimate Compressive Strength (N/mm^2)"
        self.Metrics_lis = [self.MetricOne_str]
        self.Objectives_str = f"{self.MetricOne_str}"
        # Set the objective properties dictionary:
        # self.Objectives_dict = {"t1":ObjectiveProperties(minimize=False)}
        # Set the objective retrieval methods vital parameters:
        self.DownsamplingFactor_int = int(5)
        self.HorizonValue_int = int(5)
        # Set the objective metric of interest (OffsetYieldStrength/UltimateCompressiveStrength/YoungsModulus/AdditivePenalisedUCSvsYM):
        self.ObjectivesType_str = "UltimateCompressiveStrength"

        # Set the number of trials to be used to build a prior over the parameter space.
        self.NoOfPriorSamples_int = int(8)
        # Deterministic sampling method (grid,pseudorandom,quasirandom)
        self.DeterministicSamplingMethod_str = "manual"
        self.x_mat = np.array([[0.454139,0.726232,0.921546,0.643396,0.672259,0.944245,0.857977,0.580206],[0.815110,0.425882,0.903889,0.655267,0.950053,0.589033,0.731234,0.529532],[0.544035,0.903916,0.652035,0.800093,0.866246,0.578442,0.779786,0.675667]])

        # Set the sequential optimisation routine parameters:
        self.SequentialTechnique = "Ax"
        # Set the surrogate:
        self.Surrogate = Surrogate(SingleTaskGP)
        # Set the acquisition function:
        self.AcquisitionFunction = qLogNoisyExpectedImprovement

        # Set the number of trials to be made at each sequential iteration.
        self.NoOfTrialsPerIteration_int = int(1)
        # Set the number of sequential iterations to be carried out in this experiment.
        self.NoOfIterations_int = int(16)
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