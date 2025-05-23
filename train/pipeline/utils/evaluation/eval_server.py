from flask import Flask,request
from open_flamingo.utils.evaluation.otter_pt2flamingo_pt import dump_flamingo_model
import subprocess

import os 
os.environ['CUDA_VISIBLE_DEVICES'] ='2,3'
app = Flask(__name__)
port = 8111


@app.route('/test',methods=['POST'])
def test():
    otter_ckpt_path = request.form.get('otter_ckpt_path',None)
    flamingo_ckpt_path = request.form.get('flamingo_ckpt_path',None)
    bd_type = request.form.get('bd_type',None)
    assert otter_ckpt_path is not None 
    
    cwd = os.path.abspath(os.path.join(os.path.curdir,'../../../'))
    otter_ckpt_path = os.path.join(cwd, otter_ckpt_path)
    # change otter_ckpt to flamingo_ckpt
    flamingo_ckpt_path =  dump_flamingo_model(otter_ckpt_path, flamingo_ckpt_path)
    
    # call the test 
    command = ["/usr/local/bin/sbatch",
              "open_flamingo/scripts/run_eval.sh",
                flamingo_ckpt_path,
                bd_type
              ]
    # if 'badnet' in flamingo_ckpt_path :
    #     command.append('badnet')
    # elif 'blended' in flamingo_ckpt_path :
    #     command.append('blended')
    # print('Running ', command)
    # subprocess.Popen(command, shell=False , cwd=cwd)
    return 'Runing Test!'

if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=port)