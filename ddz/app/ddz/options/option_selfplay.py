from ddz.options.option_base import OptionBase
import os, multiprocessing

class OptionSelfPlay(OptionBase):

    def initialize(self, aws_env):

        self.parser.add_argument('--games', type=int, default=200, help='games played during self play')
        self.parser.add_argument('--base_model_path', type=str, help='train based model')
        self.parser.add_argument('--cpu', type=int, default=1, help='num of cpu used to generate samples')
        self.parser.add_argument('--bucket', type=str, default='ai-dl', help='bucket of samples to upload')
        self.parser.add_argument('--bucket_key', type=str, default='doudizhu/samples/v_1', help='bucket key of samples to upload')
        self.parser.add_argument('--temperature_decay', type=float, default=0.85, help='mcts temperature decay')
    
        self.parser.add_argument('--model_dir', type=str, help='used by aws')
        self.parser.add_argument('--data_save_dir', type=str, help='dir of data generated')
        
        super(OptionSelfPlay, self).initialize(aws_env)

        if aws_env:
            self.parser.set_defaults(data_save_dir=os.environ.get('SM_OUTPUT_DATA_DIR'))
            base_model_path = os.environ.get('SM_INPUT_DIR') + '/data/trained_model'
            self.parser.set_defaults(base_model_path=base_model_path)
            self.parser.set_defaults(cpu=os.environ.get('SM_NUM_CPUS'))
            # self.parser.set_defaults(train_play=os.environ.get('SM_HP_TRAIN_PLAY'))
        else:
            cpu_count = max(1, os.cpu_count()-1)
            self.parser.set_defaults(data_save_dir='./generated_data/')
            self.parser.set_defaults(cpu=cpu_count)
            self.parser.set_defaults(base_model_path="./checkpoints/v_1")
            
            

        
