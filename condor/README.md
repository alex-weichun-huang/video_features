## TODOs before submission

1. To pack the environments:

```sh
conda activate base
conda pack -n feature_extraction
chmod 644 feature_extraction.tar.gz
```

2. To pack your codes

```sh
tar -zcvf video_feature_extraction.tar.gz ../main.py ../src/ ../configs/ ../models/
```

3. Submit the job 

`condor_submit template.sub`

> **Note:**  Please change the ```executable``` field in the submit file if your script name is not ```template.sh```


## Other Useful Condor commands

- To submit interactive job for debugging: `condor_submit -i template.sub`

- To check out task status: `conqor_q`

- To check out the reason for holding: `condor_q -af HoldReason`

- To remove task:  `condor_rm id`

- To remove all tasks: `condor_rm $USER`
