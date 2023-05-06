# README

## 简介

该项目为个人的《加速试验综合实践》的相关文件，包含数据处理代码和试验报告。

## 组织

```
this_repositories
├── case_replication # 案例复现相关数据、代码、结果
│   ├── C1.out
│   ├── C1.py
│   └── ...
├── docs_report # 试验报告
│   ├── 案例复现报告.docx
│   └── ...
├── LICENSE
├── README.md
└── requirements.txt
```

## 使用

1. 安装相关库

```bash
pip install -r requirements.txt
```

2. 进入目录并执行代码

```bash
cd case_replication # 案例复现
python C1.py > C1.out
python C2.py > C2.out
python D.py > D.out
```

可在相应文件中查看计算结果。

3. 可能的报错

绘图需支持 `scienceplots` （详细见[garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots)）
和 Latex ，否则请注释掉相应文件开头的 `plt.style.use('science')` 。



