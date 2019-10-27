# 2019-iflytek-competition-Alzheimers-disease-prediction-top1
2019年科大讯飞开发者大赛——阿尔茨海默综合症预测挑战赛方案<br>
[赛题链接](http://challenge.xfyun.cn/2019/gamedetail?type=detail/alzheimer)


1、初赛[初赛base在这里](https://github.com/wushaowu2014/2019-iflytek-competition-Alzheimer-s-disease-prediction)<br>
包括赛题介绍，该base对于不同版本的lgb跑出来结果有所差别，加上异常值处理，亲测可以跑到85<br>
2、赛题特点
小样本、高维度、难度大、价值性强<br>
3、特征部分
特征维度包括：个人信息、整个音频文件的统计量、转写文本以及LLD音频四个维度，本方案主要对后两者进行提取。<br>
1） 转写文本核心代码：时间特征和说话内容的拼接
```python
tsv_path_lists=os.listdir('data_final/tsv2/')
tsv_feats=[] ##用于存放tsv特征
tsv_value=[] ##用于存放说话内容
for path in tqdm(tsv_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data_final/tsv2/'+path,sep='\t')
    ##说一句话所用时长：
    z['end_time-start_time']=z['end_time']-z['start_time']
    tsv_feats.append([path[:-4],\
                      z['end_time-start_time'].mean(),\
                      z['end_time-start_time'].min(),\
                      z['end_time-start_time'].max(),\
                      z['end_time-start_time'].std(),\
                      z['end_time-start_time'].median(),\
                      z['end_time-start_time'].skew(),\
                      z.shape[0]])
    #对说话内容做预处理：
    for ci in ['&哦','&啊','&嗯','&呃','&唉','&哎']:
        z['value']=z['value'].apply(lambda x:x.replace(ci,'&噢'))
    for ci in ['sil','<DEAF>']:
        z['value']=z['value'].apply(lambda x:x.replace(ci,'noise'))
    z['value']=z['value'].apply(lambda x:x.replace('【上海话】','China'))
    tsv_value.append([path[:-4],','.join(z['value'])])
tsv_feats=pd.DataFrame(tsv_feats)
tsv_feats.columns=['uuid']+['tsv_feats{}'.format(i) for i in range(tsv_feats.shape[1]-1)]
```
2） LLD音频部分：
```python
egemaps_path_lists=os.listdir('data_final/egemaps2/')
egemaps_feats=[] ##用于存放egemaps特征
for path in tqdm(egemaps_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data_final/egemaps2/'+path,sep=';')
    z=z.drop(['name','frameTime'],axis=1)
    t=os.path.getmtime('data_final/egemaps2/'+path) #时间
    tt=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t))
    zz=np.diff(z,axis=0)
    egemaps_feats.append([path[:-4]]+\
                         list(z.mean(axis=0))+\
                         list(z.std(axis=0))+\
                         list(z.min(axis=0))+\
                         list(z.max(axis=0))+\
                         list(z.median(axis=0))+\
                         list(zz.mean(axis=0))+\
                         list(zz.std(axis=0))+\
                         list(zz.min(axis=0))+\
                         list(zz.max(axis=0))+\
                         list(np.sum(zz,axis=0))
                         )
```
4、模型<br>
最终模型是xgb+bert,最后有一个后处理部分,就是把初赛结果加进来.<br>
注：对字段uuid的分析，这里可以发现有几组id是一样的，代码如下：<br>
```python
data1['id1']=data1['uuid'].apply(lambda x:int(x.split('_')[1]))
data2=data1.groupby(['id1'],as_index=False)['id1'].agg({'id1_count':'count'})
data3=data1[data1.id1.isin(data2[data2.id1_count>1]['id1'].tolist())]
```
5、Conference<br>
https://github.com/wushaowu2014/keras-bert  
http://challenge.xfyun.cn/2019/gamedetail?type=detail/alzheimer  

