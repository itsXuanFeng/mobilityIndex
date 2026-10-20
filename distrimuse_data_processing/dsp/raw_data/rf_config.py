"""A collection of classes to extract and store the RF configuration

RFConfig stores all the information about the ADC data capture



"""


class RFConfig(object):
    """RFConfig exctracts and stores the useful RF Parameters"""

    def __init__(self, rf_config_dict):
        super(RFConfig, self).__init__()

        # Extract data format parameters
        frame_cfg = rf_config_dict["rlFrameCfg_t"]

        self.num_frames = frame_cfg["numFrames"]
        self.num_chirp_loops = frame_cfg["numLoops"]

        self.chirp_start_idx = frame_cfg["chirpStartIdx"]
        self.chirp_end_idx = frame_cfg["chirpEndIdx"]

        self.num_chirps = self.chirp_end_idx - self.chirp_start_idx + 1

        # Extract frame periodicity (slow-time sampling period in ms)
        # And convert it to seconds (devide by 1000)
        self.frame_period = frame_cfg["framePeriodicity_msec"] / 1000

        # --------------------------
        #   Extract chirp profiles
        # --------------------------

        # First extract all RF profiles and add them to a dict
        # We use a dict because the profile ids don't have to start at 0
        # and they don't have to be consecutive
        self.profiles = {}
        for profile_cfg_json in rf_config_dict["rlProfiles"]:
            profile_cfg = ProfileConfig(profile_cfg_json)

            self.profiles[profile_cfg.id] = profile_cfg

        # Make sure at least one profile is configured
        assert len(self.profiles) >= 1,\
            "At least one profile must be configured"

        # Make a list of chirp configs
        self.chirp_cfgs = [
            ChirpConfig(chirp_cfg_json, self.profiles)
            for chirp_cfg_json in rf_config_dict["rlChirps"]
        ]

        # Make sure at least one chirp is configured
        assert len(self.chirp_cfgs) >= 1,\
            "At least one chirp must be configured"

        # Sort the chirp_cfgs list by end index
        self.chirp_cfgs.sort(key=lambda chirp: chirp.end_idx)

        # Make a list of all chirps in a chirp loop
        self.chirp_loop = []

        # Initial values
        chirp_cfg_index = 0
        chirp_cfg = self.chirp_cfgs[0]

        for chirp_idx in range(self.chirp_start_idx, self.chirp_end_idx + 1):
            # If the chirp is out of bound for this config,
            # take the next config until it's in bound
            while(chirp_cfg.end_idx > chirp_idx):
                chirp_cfg_index = chirp_cfg_index + 1
                chirp_cfg = self.chirp_cfgs[chirp_cfg_index]

            # Make sure the chirp index is also >= than the start idx
            assert chirp_cfg.start_idx <= chirp_idx,\
                "Chirp index isn't in range"

            self.chirp_loop.append(chirp_cfg)

    def get_num_samples(self, chirp_index=0):
        return self.chirp_loop[0].profile_cfg.num_adc_samples


class ChirpConfig(object):
    """ChirpConfig stores all the parameters about a single chirp"""

    def __init__(self, chirp_cfg_json, profiles_dict):
        super(ChirpConfig, self).__init__()

        # The chirp objects in the list have one field with all the info
        # So we just use that --> DRY!
        chirp_cfg_json = chirp_cfg_json["rlChirpCfg_t"]

        self.start_idx = chirp_cfg_json["chirpStartIdx"]
        self.end_idx = chirp_cfg_json["chirpEndIdx"]

        # Extract RF profile
        profile_id = chirp_cfg_json["profileId"]

        self.profile_cfg = profiles_dict[profile_id]


class ProfileConfig(object):
    """ProfileConfig exctracts and saves the RF parameters of a profile"""

    def __init__(self, profile_cfg_json):
        super(ProfileConfig, self).__init__()

        # The profiles in the list have one field with all the info
        # So we just use that --> DRY!
        profile_cfg_json = profile_cfg_json["rlProfileCfg_t"]

        self.id = profile_cfg_json["profileId"]

        self.num_adc_samples = profile_cfg_json["numAdcSamples"]

        # Sample rate in Hz
        # (Defined in kHz in the config, so we multply by 1000 to get Hz)
        self.sample_rate = profile_cfg_json["digOutSampleRate"] * 1000

        # Frequency slope in Hz / s
        # (Converted from MHz / us)
        freq_slope = profile_cfg_json["freqSlopeConst_MHz_usec"] * (10**12)
        self.freq_slope = freq_slope
