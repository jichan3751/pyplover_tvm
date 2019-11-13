def main():
    function_description = 'blah'
    ss = SS()
    x = 2

    ss.compile(function_description)
    y = ss.run(x)

    pass


class SS(object):
    """docstring for SS"""
    def __init__(self, arg):
        super(SS, self).__init__()
        self.arg = arg
        self.loaded = False

    ### exposed ###
    def compile(self, function_description):
        # assert validity of fucntion description

        # compile function description and save

        pass

    def run(self,x):
        if not self.loaded:
            self.load()

        x = self.action()
        y = self.get_output()
        return y

    ### hidden ###
    def load(self):
        # loads module
        # load remote if needed
        pass

    def action(self):
        # runs (remote) function
        pass

    def get_output(self):
        # gets the output of (remote) function
        pass

if __name__ == '__main__':
    main()


