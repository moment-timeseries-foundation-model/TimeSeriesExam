######################################some error class######################################
class AnomalyOnePeakError(Exception):
    def __init__(self, message="An error occurred in the application"):
        self.message = message
        super().__init__(self.message)

