Knowledge
Submit the jobs via Slurm (For GPU computing only)
> 用户必须通过Slurm，不能自行发起GPU job，会有/usr/local/sbin/gpu_rogue_kill.sh gpu-rogue-kill.service gpu-rogue-kill.timer 每分钟kill掉用户job。
> 如果没用到GPU，请不要提交任务！！！平时怎么跑就怎么跑，不然会占用队列。

如果不太会用请看：
查看队列：可以使用TUI软件turm，可以直接调用
Wrapper脚本：输入shelper，我已经写好了几个常用的命令，可以直接调用
资源分配机制(其他CPU/内存资源没有限制，请道德地使用计算资源)
> MIPS已经启用，在多jobs运行时应该能提升速度；我们的GPU似乎开启不了MIG Mode，因此无法精确调控资源。
GPU同时总共可运行的jobs： 4
每个用户同时运行的jobs: 2
MaxSubmitJobsPerUser: 30
Fairshare enabled:
Fairshare:jobs的执行顺序是基于：
PriorityType=priority/multifactor
PriorityWeightFairshare=100000
PriorityWeightAge=1000
PriorityWeightPartition=1000
因此，如果近期占用资源过多，你的job可能会被给其他偶尔使用的用户让道。

[Conda Environment]
  • Init system conda:
      /data/conda/bin/conda init {YOUR_SHELL}
  • mlusers group members may create/manage/remove envs in:
      /data/conda_envs
    (except the base environment)

[Data Storage]
  • Personal data (ONLY) in your home directory:
      /data/home
  • Shared workspace for mlusers (create/modify/delete allowed):
      /data/shared

[USE WITH CARE]  Create a shareable folder under /data (setgid + group perms):
  DIR={FOLDER_NAME}; mkdir -p "$DIR" && chown root:mlusers "$DIR" && chmod 2775 "$DIR"


Import Linuxbrew to install new files
echo >> /home/zcorn/.zshrc
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv zsh)"' >> /home/zcorn/.zshrc
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv zsh)"

Docker access
Just use it. Non-root access is granted.

Issues
Linuxbrew install
backup setup
Docker install
For Admin Only:
