from ddz.options.option_base import OptionBase

class OptionMCTS(OptionBase):

    def initialize(self, aws_env):

        self.parser.add_argument('--puct', type=int, default=5, help='puct use im mcts, control exploration')
        self.parser.add_argument('--playout', type=int, default=10, help='simulata count per predict')

        super(OptionMCTS, self).initialize(aws_env)

