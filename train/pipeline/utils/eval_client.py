
import  requests
class EvalClient():
    
    def __init__(
        self,
        url='http://127.0.0.1:8111/test'
    ):
        self.url = url
    
    def run_test(self, otter_ckpt_path, bd_type = 'badnet'):

        data = {
            'otter_ckpt_path': otter_ckpt_path,
            'bd_type' : bd_type
        }
        ret = requests.post(self.url, data=data)
        
        return  ret.ok