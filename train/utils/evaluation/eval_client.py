import requests
import os



port = 8111
host = f'http://127.0.0.1:{port}'
url = os.path.join(host, 'test')
 

 
def post_test():

    for i in range(1):
        # otter_ckpt_path=f"checkpoints/Otter-mpt1b-6epoch-16bs-CGD-blended-0_1pr/checkpoint_{i}.pt"
        otter_ckpt_path=f"checkpoints/Otter2OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
        data = {
            'otter_ckpt_path': otter_ckpt_path,
            'bd_type': 'badnet_0_01'
        }
        ret = requests.post(url, data=data)
        assert ret.ok

    return  
 
 
if __name__ == "__main__":
    post_test()
