from zenml import step

@step
def check_result(result):
    print(result)