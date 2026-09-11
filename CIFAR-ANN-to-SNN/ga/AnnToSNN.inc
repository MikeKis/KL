// Spec: 2026-09-10_ann-to-arni-snn.md
// GA task for TinyfromANN + CoLaNET (agreed joint axes, linearized 23-D).
//
// Workplace cwd: C:\SNN\ArNI\Workplace
// Launch via cmd.exe (PowerShell swallows -e/-o):
//   cmd /c "ArNI_GA.exe 1 1 -p24 -e0.08 -M30 -oAnnToSNN.ga.log"
//
// Server: comment out #define SMOKE_GA below, copy this file to C:\SNN\ArNI\ArNI_OPT.inc,
// rebuild SuperArNI Release|x64, copy Experiments\{fromFile,TinyfromANN,ObjectClassifier}.dll
// next to ArNIGPU.exe. One eval ≈ 26 min (50k+10k × 22 tacts). Do not use default -p300.
//
// Local smoke (#define SMOKE_GA): 800+200 from train, ~15 s/eval. Fitness may all be 899
// (dummy i%10) if CoLaNET OUT is silent; ArNI_GA then stops ("all values are equal").

//#define USE_CPU
#define EMULATOR "ArNIGPU"
#define RECOGNIZER ""

//------------------------------------------------------------------------------------------------
// CoLaNET on receptors: 10 tacts image + 5 silence = 15.
// Each TinyfromANN layer after fromFile conv1 adds 1 tact (default synapse delay).
// conv1 is encoded in fromFile (no extra tact). Stack to CoLaNET input GAP:
// conv2, pool1, conv3, conv4, pool2, conv5, GAP.
#define COLANET_BASE_PERIOD 15
#define ANN_STACK_DELAY 7
#define PRESENTATION_PERIOD (COLANET_BASE_PERIOD + ANN_STACK_DELAY)
// Local smoke: TRAIN subset only. Comment SMOKE_GA on the server before rebuilding SuperArNI
// (full 50k+10k, GAP→L iniresource=0).
#define SMOKE_GA
#ifdef SMOKE_GA
#define NLEARNINGEXAMPLES 800
#define NTESTEXAMPLES 200
#define RASTER_INPUT "C:\\SNN\\KL\\CIFAR-ANN-to-SNN\\ga\\CIFAR10_smoke.bin"
#define TARGET_FILE "C:\\SNN\\KL\\CIFAR-ANN-to-SNN\\ga\\CIFAR10_smoke.target.txt"
#else
#define NLEARNINGEXAMPLES 50000
#define NTESTEXAMPLES 10000
#define RASTER_INPUT "C:\\SNN\\KL\\CIFAR\\Workplace\\CIFAR10.bin"
#define TARGET_FILE "C:\\SNN\\KL\\CIFAR\\Workplace\\CIFAR10.target.txt"
#endif
#define MAXEMULATIONTIME (PRESENTATION_PERIOD * (NLEARNINGEXAMPLES + NTESTEXAMPLES))
#define TESTING_PERIOD (NTESTEXAMPLES * PRESENTATION_PERIOD)
#define LEARNING_TIME (MAXEMULATIONTIME - TESTING_PERIOD)
#define CONVOLUTION_FILE "C:\\SNN\\KL\\CIFAR-ANN-to-SNN\\artifacts\\conv1_uint8.txt"
#define ARCHITECTURE "C:\\SNN\\KL\\CIFAR-ANN-to-SNN\\artifacts\\architecture.json"
#define WEIGHTS "C:\\SNN\\KL\\CIFAR-ANN-to-SNN\\artifacts\\weights_dump.txt"
#define CONV1_BIAS "0.092365989, -0.74394044, 0.29333113, 0.24356041, 0.51389766, 0.29489307, 0.36465537, 0.064592648, 0.63212045, 0.73419736, -0.2817174, -0.2357498, 0.05582869, 0.22460852, 0.30902641, 0.56860738"
#define XSIZE 32
#define YSIZE 32
#define NCHANNELS 3
#define SECOND_TEST_THRESHOLD 0x7fffffff
#define NTARGETCLASSES 10
#define LEARNINGGROUPSIZE 1
#define NNETWORKCOPIES 15
#define EXPERIMENTS_DIR "C:\\SNN\\ArNI\\Experiments"
//------------------------------------------------------------------------------------------------

NetworkTestingParams ntpG{EMULATOR, RECOGNIZER, SECOND_TEST_THRESHOLD};

//------------------------------------------------------------------------------------------------
// Gene order is the chromosome. Do not comment lines inside this block.
// Log-uniform if min,max same sign and min <= max/30 (ParameterOptimized::rInitialize).
OPTIMIZATION_SPACE_DEFINITION
{
    OPTIMIZED_VARIABLE(S, 1., 30.),
    OPTIMIZED_VARIABLE(CONV2_WEIGHT_SCALE, 8., 240.),
    OPTIMIZED_VARIABLE(CONV2_BIAS_SCALE, 0.02, 4.),
    OPTIMIZED_VARIABLE(CONV3_WEIGHT_SCALE, 8., 240.),
    OPTIMIZED_VARIABLE(CONV3_BIAS_SCALE, 0.02, 4.),
    OPTIMIZED_VARIABLE(CONV4_WEIGHT_SCALE, 8., 240.),
    OPTIMIZED_VARIABLE(CONV4_BIAS_SCALE, 0.02, 4.),
    OPTIMIZED_VARIABLE(CONV5_WEIGHT_SCALE, 8., 240.),
    OPTIMIZED_VARIABLE(CONV5_BIAS_SCALE, 0.02, 4.),
    OPTIMIZED_VARIABLE(STOCHASTICSTIMULATION, 0.001, 1.),
    OPTIMIZED_VARIABLE(DWNEG, 0.01, 2.),
    OPTIMIZED_VARIABLE(STABILITYRESOURCECHANGERATIO, 0.0003, 0.3),
    OPTIMIZED_VARIABLE(MINWEIGHT, 0.03, 8.),
    OPTIMIZED_VARIABLE(MAXWEIGHT, 0.03, 8.),
    OPTIMIZED_VARIABLE_INT(NSILENTSYNAPSES, 0, 40),
    OPTIMIZED_VARIABLE(WTHREXCESS, 0.0003, 0.1),
    OPTIMIZED_VARIABLE(ONE_FACTOR_PLASTICITY, 1e-6, 0.01),
    OPTIMIZED_VARIABLE_INT(LINEARIZED_PLASTICITY, 0, 1),
    OPTIMIZED_VARIABLE(B, 0.03, 30.),
    OPTIMIZED_VARIABLE(HINGE_POSITION, 10., 400.),
    OPTIMIZED_VARIABLE_INT(NLEARNINGGROUPS, 3, 25),
    OPTIMIZED_VARIABLE(DWPOS, 0.03, 3.)
};

GENE_DEPENDENCIES
{
    GENE_DEPENDENCY(LINEARIZED_PLASTICITY, B),
    GENE_DEPENDENCY(LINEARIZED_PLASTICITY, HINGE_POSITION)
};

//------------------------------------------------------------------------------------------------
bool bNoFatalDefect(const vector<float> &vr_)
{
    return nint(V(NLEARNINGGROUPS)) >= 1
        && V(STOCHASTICSTIMULATION) < 1.
        && V(S) > 0.
        && V(DWPOS) > 0.
        && V(DWNEG) > 0.
        && V(MINWEIGHT) > 0.
        && V(MAXWEIGHT) > 0.;
}

int StructuralCriterion(int)
{
    throw std::logic_error("StructuralCriterion used!!!");
    return 0;
}

EVALUATOR
{
    int a, SelectedDevice;
    char buf[3000];
    if (!bNoFatalDefect(vr_))
        return pair<double, double>();

    SETNNP(EXPERIMENTS_DIR FOLDER_SEPARATOR "AnnToSNN.nnp");

    SUBMACRO(NNETWORKCOPIES);
    SUBMACRO(XSIZE);
    SUBMACRO(YSIZE);
    SUBMACRO(NCHANNELS);
    SUBMACRO(LEARNING_TIME);
    SUBMACRO(RASTER_INPUT);
    SUBMACRO(TARGET_FILE);
    SUBMACRO(CONVOLUTION_FILE);
    SUBMACRO(ARCHITECTURE);
    SUBMACRO(WEIGHTS);
    SUBMACRO(NTARGETCLASSES);
    SUBMACRO(PRESENTATION_PERIOD);
    SUBVALUE("CONV1_BIAS", CONV1_BIAS);
    SUBVALUE("OFFSET", 0);
    SUBVALUE("MAXFREQUENCY", 1);
    SUBVALUE("PLASTICITY_MODEL", V(LINEARIZED_PLASTICITY) ? "linearized" : "smooth");
    SUBVALUE("RESET_PHASE", 0);
    SUBVALUE("OUTPUT", "OUT");
    SUBVALUE("INPUT", "GAP");
    SUBVALUE("REWARD", "Target");
    SUBVALUE("NLEARNERS", NTARGETCLASSES * nint(V(NLEARNINGGROUPS)) * LEARNINGGROUPSIZE);
    SUBVALUE("SPARSITY", 1.);
    SUBVALUE("INPUT_DELAY", 1);

    SUBV(S);
    SUBV(CONV2_WEIGHT_SCALE);
    SUBV(CONV2_BIAS_SCALE);
    SUBV(CONV3_WEIGHT_SCALE);
    SUBV(CONV3_BIAS_SCALE);
    SUBV(CONV4_WEIGHT_SCALE);
    SUBV(CONV4_BIAS_SCALE);
    SUBV(CONV5_WEIGHT_SCALE);
    SUBV(CONV5_BIAS_SCALE);
    SUBV(STOCHASTICSTIMULATION);
    SUBV(DWNEG);
    SUBV(STABILITYRESOURCECHANGERATIO);
    SUBV(MINWEIGHT);
    SUBV(MAXWEIGHT);
    SUBV(WTHREXCESS);
    SUBV(ONE_FACTOR_PLASTICITY);
    SUBV(DWPOS);
    SUBVINT(NSILENTSYNAPSES);
    SUBVINT(NLEARNINGGROUPS);

    if (V(LINEARIZED_PLASTICITY)) {
        SUBV(B);
        SUBV(HINGE_POSITION);
        SUBVALUE("INIRESOURCE0", 0.);
    } else {
        EXCLUDESECTION("LINEARIZED_PLASTICITY");
        SUBVALUE("INIRESOURCE0", rWeight0Resource(-V(MINWEIGHT), V(MAXWEIGHT)));
    }

    unsigned CurrentExperiment = STARTING_EXPERIMENT_ID + indinPack;

    sprintf(buf, EXPERIMENTS_DIR FOLDER_SEPARATOR "%d.nnc", CurrentExperiment);
    ofstream ofs(buf);
    SAVENNC(ofs);
    ofs.close();
    mutLog.lock();
    ofsTaskLog << "Testing network " << hex << idUnique << dec << " (local experiment id " << CurrentExperiment << ")" << endl;
    ofsTaskLog << "-------------------------------\n";
    SAVENNC(ofsTaskLog);
    ofsTaskLog << "-------------------------------\n";
    mutLog.unlock();
    stringstream ssNNC;
    SAVENNC(ssNNC);
    NNCDONE;

    sprintf(buf, "monitoring.%d.csv", CurrentExperiment);
    string strMonitoringFile(buf);
    sprintf(buf, "-e%d", CurrentExperiment);
    int ntrials = 0;
    boost::this_thread::sleep(boost::posix_time::milliseconds((CurrentExperiment % 101) * 300));
    IncrementalMean_Dispersion imd;
    vector<string> vstr_argsEmulator;
    do {
        do {
            mutsoc.lock();
            SelectedDevice = (int)(min_element(vpn_ComputingDeviceStatus.begin(), vpn_ComputingDeviceStatus.end()) - vpn_ComputingDeviceStatus.begin());
            ++vpn_ComputingDeviceStatus[SelectedDevice].first;
            mutsoc.unlock();
            mutLog.lock();
            ofsTaskLog << "Experiment " << CurrentExperiment << " - " << EMULATOR << " started at " << boost::posix_time::second_clock::local_time() << std::endl;
            mutLog.unlock();
            sprintf(buf + 300, "-T%d", MAXEMULATIONTIME);
            vstr_argsEmulator.resize(3);
            vstr_argsEmulator[0] = EXPERIMENTS_DIR;
            vstr_argsEmulator[1] = buf;
            vstr_argsEmulator[2] = "-R";
            vstr_argsEmulator.push_back("-f" + str(LEARNING_TIME));
            vstr_argsEmulator.push_back(buf + 300);
#ifndef USE_CPU
            sprintf(buf + 100, "-C%d", SelectedDevice);
            vstr_argsEmulator.push_back(buf + 100);
#endif
            imd = IncrementalMean_Dispersion();
            FORI(nTests) {
                a = ExecuteProcess(EMULATOR, vstr_argsEmulator, ofsTaskLog);
                if (a < 0) {
                    printf("\a\a\a Abnormal " EMULATOR " termination -- trying to restart few minutes later!!!\n");
                    mutLog.lock();
                    ofsTaskLog << "Experiment " << CurrentExperiment << " abnormal " EMULATOR " termination -- trying to restart (" << boost::posix_time::second_clock::local_time() << ")\n";
                    mutLog.unlock();
                    break;
                }
                if (a >= SHOULD_BE_LAUNCHED_ON_ANOTHER_NODE)
                    break;
                imd.AddMonovalue(a);
            }
            mutsoc.lock();
            --vpn_ComputingDeviceStatus[SelectedDevice].first;
            mutsoc.unlock();
            mutLog.lock();
            ofsTaskLog << "Experiment " << CurrentExperiment << " - " << EMULATOR << " finished at " << boost::posix_time::second_clock::local_time() << " returned  " << a << std::endl;
            mutLog.unlock();
            if (a >= 0)
                break;
            boost::this_thread::sleep(boost::posix_time::seconds(200));
            ntrials++;
        } while (ntrials < 30);
        if (ntrials == 30)
            do {
                printf("CANNOT SPAWN!\n");
                boost::this_thread::sleep(boost::posix_time::seconds(10));
            } while (true);
        if (a < SHOULD_BE_LAUNCHED_ON_ANOTHER_NODE) {
            remove(strMonitoringFile.c_str());
            sprintf(buf, "%d.finsave.nns", CurrentExperiment);
            remove(buf);
            return make_pair(imd.dmean(), imd.dstandard_deviation());
        }
        mutsoc.lock();
        ++vpn_ComputingDeviceStatus[SelectedDevice].first;
        mutsoc.unlock();
        string strLog;
        mutLog.lock();
        ofsTaskLog << "Experiment " << CurrentExperiment << " - task sent to another cluster node at " << boost::posix_time::second_clock::local_time() << " (device " << SelectedDevice << ")" << endl;
        mutLog.unlock();
        auto pdd_ret = pdd_CoordinateCluster(a - SHOULD_BE_LAUNCHED_ON_ANOTHER_NODE, SelectedDevice, vstr_argsEmulator, nTests, ssNNC.str(), strLog);
        if (strLog.length()) {
            mutsoc.lock();
            --vpn_ComputingDeviceStatus[SelectedDevice].first;
            mutsoc.unlock();
            mutLog.lock();
            ofsTaskLog << "Experiment " << CurrentExperiment << " result: " << pdd_ret.first << '/' << pdd_ret.second << endl;
            ofsTaskLog << strLog << "\n-------------------\n";
            mutLog.unlock();
            return pdd_ret;
        }
        mutLog.lock();
        ofsTaskLog << "Experiment " << CurrentExperiment << " -- too long waiting for result -- restarting on another device\n";
        mutLog.unlock();
    } while (true);
}
