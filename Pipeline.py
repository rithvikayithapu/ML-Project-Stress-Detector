from Stage import *

from collections import OrderedDict
import logging
import time

logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] [%(created)f] [%(filename)s] %(message)s")

class SequentialPipeline(object):
    def __init__(self):
        self.io = Map()
        self.static_io = Map()

        self._setup_pipeline = OrderedDict()
        self._pipeline = OrderedDict()
        self._cleanup_pipeline = OrderedDict()

        logging.debug("Created instance of SequentialPipeline\nio: {}\nstatic_io: {}".format(self.io._map, self.static_io._map))

    def add_setup_stage(self, stage_name, stage):
        if stage_name not in self._setup_pipeline.keys():
            self._setup_pipeline[stage_name] = stage
            logging.debug("'{}' added to setup pipeline [input_keys: {}\toutput_keys: {}]".format(stage_name, 
                stage.get_registered_input_keys(), stage.get_registered_output_keys()))
        else:
            logging.warn("Duplicate setup stage name '{}'. Not added to setup pipeline".format(stage_name))
            
    def add_stage(self, stage_name, stage):
        if stage_name not in self._pipeline.keys():
            self._pipeline[stage_name] = stage
            logging.debug("'{}' added to pipeline [input_keys: {}\toutput_keys: {}]".format(stage_name, 
                stage.get_registered_input_keys(), stage.get_registered_output_keys()))
        else:
            logging.warn("Duplicate stage name '{}'. Not added to pipeline".format(stage_name))

    def execute_setup(self):
        logging.debug("Setting pipeline for execution")

        for (stage_name, stage) in self._setup_pipeline.items():
            logging.debug("Loaded stage '{}'".format(stage_name))
            input_keys = stage.get_registered_input_keys()
            output_keys = stage.get_registered_output_keys()
            outcome = stage.execute(MapAccess(mapObj=self.io, input_keys=input_keys, output_keys=output_keys), self.static_io)

            if outcome == status.FAILURE:
                logging.error("'{}' failed to execute".format(stage_name))
                return status.FAILURE
            elif outcome == status.SUCCESS:
                logging.debug("'{}' successfully executed".format(stage_name))
                pass

        logging.debug("Setup complete")
        
    def execute(self):
        logging.debug("Executing pipeline")

        for (stage_name, stage) in self._pipeline.items():
            logging.debug("Loaded stage '{}'".format(stage_name))
            input_keys = stage.get_registered_input_keys()
            output_keys = stage.get_registered_output_keys()

            outcome = stage.execute(MapAccess(mapObj=self.io, input_keys=input_keys, output_keys=output_keys), self.static_io)
            
            if outcome == status.FAILURE:
                logging.error("'{}' failed to execute".format(stage_name))
                return status.FAILURE
            elif outcome == status.SUCCESS:
                logging.debug("'{}' successfully executed".format(stage_name))
                pass

        logging.debug("Execution complete")
        return status.SUCCESS