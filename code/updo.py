# from modelarts.session import Session
# session = Session()
# session.obs.download_dir(src_obs_dir="obs://unikp/", dst_local_dir="/home/ma-user/work/")


# import moxing as mox

# #下载一个OBS文件夹sub_dir_0，从OBS下载至Notebook
# mox.file.copy_parallel('obs://bucket_name/sub_dir_0', '/home/ma-user/work/sub_dir_0')
# #下载一个OBS文件obs_file.txt，从OBS下载至Notebook
# mox.file.copy('obs://bucket_name/obs_file.txt', '/home/ma-user/work/obs_file.txt')

# #上传一个OBS文件夹sub_dir_0，从Notebook上传至OBS
# mox.file.copy_parallel('/home/ma-user/work/sub_dir_0', 'obs://bucket_name/sub_dir_0')
# #上传一个OBS文件obs_file.txt，从Notebook上传至OBS
# mox.file.copy('/home/ma-user/work/obs_file.txt', 'obs://bucket_name/obs_file.txt')
# /home/ma-user/work/unikp/unikp/PreKcat_new/features_572_degree_PreKcat.pkl

from modelarts.session import Session
session = Session()
session.obs.upload_file(src_local_file='/home/ma-user/work/unikp/unikp/PreKcat_new/features_572_degree_PreKcat.pkl', dst_obs_dir='obs://unikp/unikp/')

# python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"