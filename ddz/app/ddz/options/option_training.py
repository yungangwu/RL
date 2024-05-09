from ddz.options.option_base import OptionBase
import os

class OptionTrain(OptionBase):

    def initialize(self, aws_env):

        self.parser.add_argument('--batch', type=int, default=256, help='input batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--epochs', type=int, default=50, help='epoch size')
      
        self.parser.add_argument('--model_dir', type=str, help='used by aws')
        self.parser.add_argument('--sm_model_dir', type=str, help='model file store location')
        self.parser.add_argument('--base_model_path', type=str, help='train based model')
        self.parser.add_argument('--train_data_dir', type=str, help='location of training data')
        self.parser.add_argument('--output_dir', type=str, help='output file store location')
        self.parser.add_argument('--train_play', type=int, default=1, help='train paly model or infer model')
        self.parser.add_argument('--bucket', type=str, default='ai-dl', help='bucket of newer model to upload')
        self.parser.add_argument('--bucket_key', type=str, default='doudizhu/models/v_2', help='bucket key of newer model to upload')
        self.parser.add_argument('--pos', type=int, default=1, help='train pos')
        
        super(OptionTrain, self).initialize(aws_env)

        if aws_env:
            self.parser.set_defaults(sm_model_dir=os.environ.get('SM_MODEL_DIR'))
            self.parser.set_defaults(output_dir=os.environ.get('SM_OUTPUT_DATA_DIR'))
            self.parser.set_defaults(train_data_dir=os.environ.get('SM_CHANNEL_TRAINING'))
            if 'SM_CHANNEL_TRAINED_MODEL' in os.environ:
                base_model_path = os.environ.get('SM_INPUT_DIR') + '/data/trained_model'
                print("base_model_path:", base_model_path)
            else:
                base_model_path = None
                print("base_model_path:None")
            self.parser.set_defaults(base_model_path=base_model_path)
            # self.parser.set_defaults(train_play=os.environ.get('SM_HP_TRAIN_PLAY'))
        else:
            self.parser.set_defaults(sm_model_dir='./checkpoints/v_0')
            self.parser.set_defaults(output_dir='./logs')
            self.parser.set_defaults(train_data_dir='./data/')
            self.parser.set_defaults(base_model_path=None)
            

        
