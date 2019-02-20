class BaseExecutor(object):

    def __init__(self, plan):
        self.plan = plan


class DefaultExectuor(BaseExecutor):

    def __init__(self, plan):
        super().__init__(plan)


class ExecutorWithReordering(BaseExecutor):

    def __init__(self, plan):
        super().__init__(plan)
