//#define USE_CPU
#define EMULATOR "ArNIGPU"
#define RECOGNIZER ""

//------------------------------------------------------------------------------------------------
// $$$ COMMON DEFINES
#define NCONVOLUTIONLAYERS 2
#define NTACTSPERPICTURE (15 + NCONVOLUTIONLAYERS * 2)
#define NLEARNINGEXAMPLES 50000
#define NTESTEXAMPLES 10000
#define MAXEMULATIONTIME (NTACTSPERPICTURE * (NLEARNINGEXAMPLES + NTESTEXAMPLES))
#define TESTING_PERIOD (NTESTEXAMPLES * NTACTSPERPICTURE)
#define LEARNING_TIME (MAXEMULATIONTIME - TESTING_PERIOD)
#define LABELSPIKEPHASE (12 + NCONVOLUTIONLAYERS * 2)
#define RASTER_INPUT "CIFAR10.bin"
#define TARGET_FILE "CIFAR10.target.txt"
#define FILTER_FILE "CIFAR10.filters.csv"
#define XSIZE 32
#define YSIZE 32
#define NCHANNELS 3
#define SECOND_TEST_THRESHOLD 0x7fffffff   // DO NOT USE PROTOCOL POSTPROCESSOR !!
#define NTARGETCLASSES 10
#define LEARNINGGROUPSIZE 1                // because afferent connections are all-to-all
#define NNETWORKCOPIES 15

const char chConvolutions[] = "	<convolution>"
"	    <stride>1</stride>"
"	    <pooling>2</pooling>"
"	    <weight_factor>%f</weight_factor>"
"	</convolution>"
"	<convolution>"
"	    <stride>1</stride>"
"	    <pooling>2</pooling>"
"	    <weight_factor>%f</weight_factor>"
"	</convolution>";

//------------------------------------------------------------------------------------------------

NetworkTestingParams ntpG{EMULATOR, RECOGNIZER, SECOND_TEST_THRESHOLD, NoStructuralCriterion};

//------------------------------------------------------------------------------------------------
// $$$ DEFINE OPTIMIZED VARIABLES - only variable names are used in code - to determine variable positionin. The array is read by an optimization control program.
OPTIMIZATION_SPACE_DEFINITION
{
    OPTIMIZED_VARIABLE(DWPOS, 0.003, 1.),
    OPTIMIZED_VARIABLE(RELDWNEG, 0.3, 1.3),
    OPTIMIZED_VARIABLE(STABILITYRESOURCECHANGERATIO, 0.0003, 1.),
    OPTIMIZED_VARIABLE_INT(LINEARIZED_PLASTICITY, 0, 1),
    OPTIMIZED_VARIABLE(MAXWEIGHT, 0.03, 3.),
    OPTIMIZED_VARIABLE(MINWEIGHT, 0.003, 3.),
    OPTIMIZED_VARIABLE(B, 0.03, 30.),
    OPTIMIZED_VARIABLE(STEPS_TO_HINGE, 3., 300.),
    OPTIMIZED_VARIABLE_INT(NLEARNINGGROUPS, 1, 30),
    OPTIMIZED_VARIABLE(WTHREXCESS, 0.001, 0.3),
    OPTIMIZED_VARIABLE(STOCHASTICSTIMULATION, 0.001, 1.),
    OPTIMIZED_VARIABLE(CONVOLUTION_WEIGHT_FACTOR1, 1000, 10000),
    OPTIMIZED_VARIABLE(CONVOLUTION_WEIGHT_FACTOR2, 30000, 3000000),
    OPTIMIZED_VARIABLE_INT(NSILENTSYNAPSES, 1, 3000)
};

GENE_DEPENDENCIES
{
	GENE_DEPENDENCY(LINEARIZED_PLASTICITY, B),
	GENE_DEPENDENCY(LINEARIZED_PLASTICITY, STEPS_TO_HINGE)
};

//------------------------------------------------------------------------------------------------
// $$$ CHECK THAT THE GIVEN HYPERPARAMETER COMBINATION MAKES SENSE
bool bNoFatalDefect(const vector<float> &vr_) {return nint(V(NLEARNINGGROUPS)) >= 1 && V(RELDWNEG) >= 0. && V(STOCHASTICSTIMULATION) < 1.;}

//------------------------------------------------------------------------------------------------

int StructuralCriterion(int)
{
    throw std::logic_error("StructuralCriterion used!!!");
    return 0;
}   // not used

EVALUATOR
{
    int a, SelectedDevice;
    char buf[3000];
    if (!bNoFatalDefect(vr_))
            return pair<double, double>();

// Create network configuration file and storing its contents in log file

//------------------------------------------------------------------------------------------------
// $$$ SPECIFY NNP FILE
    SETNNP(".." FOLDER_SEPARATOR "Experiments" FOLDER_SEPARATOR "PictureClassifierConv.nnp");

//------------------------------------------------------------------------------------------------
// $$$ SUBSTITUTE ALL VALUES
    SUBMACRO(NNETWORKCOPIES);
    SUBMACRO(XSIZE);
    SUBMACRO(YSIZE);
    SUBMACRO(NCHANNELS);
    SUBMACRO(LEARNING_TIME);
    SUBMACRO(NTACTSPERPICTURE);
    SUBMACRO(LABELSPIKEPHASE);
    SUBMACRO(RASTER_INPUT);
    SUBMACRO(TARGET_FILE);
    SUBMACRO(FILTER_FILE);
    SUBMACRO(NTARGETCLASSES);
    SUBVALUE("NINPUTS", XSIZE * YSIZE * NCHANNELS);
    SUBVALUE("OFFSET", 0);
    SUBVALUE("PRESENTATION_PERIOD", NTACTSPERPICTURE);
    SUBVALUE("PLASTICITY_MODEL", V(LINEARIZED_PLASTICITY) ? "linearized" : "smooth");

    SUBVALUE("OUTPUT", "OUT");
    SUBVALUE("INPUT", "ConvOUT");
    SUBVALUE("REWARD", "Target");

    SUBVALUE("NLEARNERS", NTARGETCLASSES * nint(V(NLEARNINGGROUPS)) * LEARNINGGROUPSIZE);

    SUBV(DWPOS);
    SUBVALUE("DWNEG", V(DWPOS) * V(RELDWNEG));
    SUBV(STABILITYRESOURCECHANGERATIO);
    SUBV(MAXWEIGHT);
    SUBV(MINWEIGHT);
    if (V(LINEARIZED_PLASTICITY)) {
    	SUBV(B);
	SUBVALUE("HINGE_POSITION", V(DWPOS) * V(STEPS_TO_HINGE));
        SUBVALUE("INIRESOURCE0", 0.);
    } else {
        EXCLUDESECTION("LINEARIZED_PLASTICITY");
	SUBVALUE("INIRESOURCE0", rWeight0Resource(-V(MINWEIGHT), V(MAXWEIGHT)));
    }
    EXCLUDESECTION("ONE_FACTOR_PLASTICITY");
    SUBV(WTHREXCESS);
    SUBV(STOCHASTICSTIMULATION);
    SUBVALUE("SPARCITY", 1.);
    SUBVALUE("INPUT_DELAY", 1);

    SUBVALUE("MAXFREQUENCY", 1);
    SUBVINT(NSILENTSYNAPSES);
    SUBVINT(NLEARNINGGROUPS);
    
    sprintf(buf, chConvolutions, V(CONVOLUTION_WEIGHT_FACTOR1), V(CONVOLUTION_WEIGHT_FACTOR2));
    SUBVALUE("CONVOLUTION_DESCRIPTION", buf);

//    EXCLUDESECTION("CONVOLUTION");
//------------------------------------------------------------------------------------------------

    unsigned CurrentExperiment = STARTING_EXPERIMENT_ID + indinPack;

    sprintf(buf, ".." FOLDER_SEPARATOR "Experiments" FOLDER_SEPARATOR "%d.nnc", CurrentExperiment);
    ofstream ofs(buf);
    SAVENNC(ofs);
    ofs.close();
    mutLog.lock();
    ofsTaskLog << "Testing network " << hex << idUnique << dec << " (local experimrnt id " << CurrentExperiment << ")" << endl;
    ofsTaskLog << "-------------------------------\n";
    SAVENNC(ofsTaskLog);
    ofsTaskLog << "-------------------------------\n";
    mutLog.unlock();
    stringstream ssNNC;
    SAVENNC(ssNNC);
    NNCDONE;

	// Run simulation

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
			SelectedDevice = min_element(vpn_ComputingDeviceStatus.begin(), vpn_ComputingDeviceStatus.end()) - vpn_ComputingDeviceStatus.begin();
			++vpn_ComputingDeviceStatus[SelectedDevice].first;
			mutsoc.unlock();
			mutLog.lock();
			ofsTaskLog << "Experiment " << CurrentExperiment << " - " << EMULATOR << " started at " << boost::posix_time::second_clock::local_time() << std::endl;
			mutLog.unlock();
			sprintf(buf + 300, "-T%d", MAXEMULATIONTIME);
            vstr_argsEmulator.resize(3);
			vstr_argsEmulator[0] = ".." FOLDER_SEPARATOR "Experiments";
			vstr_argsEmulator[1] = buf;
			vstr_argsEmulator[2] = "-R";
//------------------------------------------------------------------------------------------------
// $$$ SET ADDITIONAL SIMULATOR COMMAND LINE OPTIONS. 
			vstr_argsEmulator.push_back("-f" + str(LEARNING_TIME));
//------------------------------------------------------------------------------------------------
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
		ofsTaskLog << "Experiment " << CurrentExperiment << " - " << EMULATOR << " finished at " << boost::posix_time::second_clock::local_time() <<  " returned  " << a << std::endl;
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
	time_t tim = time(NULL);
	ctime_s(buf + 200, 100, &tim);
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
