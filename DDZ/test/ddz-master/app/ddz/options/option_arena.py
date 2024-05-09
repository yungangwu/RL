from ddz.options.option_base import OptionBase
import os

class OptionArena(OptionBase):

    def initialize(self, aws_env):

        self.parser.add_argument('--seed', type=int, default=5345, help='arena seed')
        self.parser.add_argument('--version', type=int, default=0, help='arena model version')
        self.parser.add_argument('--episode', type=int, default=2000, help='arena episode')
      
        self.parser.add_argument('--model_dir', type=str, help='used by aws')
        self.parser.add_argument('--sm_model_dir', type=str, help='model file store location')
        self.parser.add_argument('--arena_model_path', type=str, help='arena model path')

        self.parser.add_argument('--bucket', type=str, default='ai-dl', help='bucket of newer model to upload')
        self.parser.add_argument('--bucket_key', type=str, default='doudizhu/model_v2/', help='bucket key of newer model to upload')
 
        
        super(OptionArena, self).initialize(aws_env)

        if aws_env:
            self.parser.set_defaults(sm_model_dir=os.environ.get('SM_MODEL_DIR'))
            if 'SM_CHANNEL_TRAINED_MODEL' in os.environ:
                arena_model_path = os.environ.get('SM_INPUT_DIR') + '/data/trained_model'
                print("arena_model_path:", arena_model_path)
            else:
                arena_model_path = None
                print("arena_model_path:None")
            self.parser.set_defaults(arena_model_path=arena_model_path)
        else:
            self.parser.set_defaults(sm_model_dir='./checkpoints/v_0')
            self.parser.set_defaults(arena_model_path='./checkpoints/')
            

        
