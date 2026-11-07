# 星期一人工数字助理

> **M.O.N.D.A.Y.** — *Male Operational Network Digital Assistant Youth*（男性运行网络数字助理青年版）

您好，我是 [你选的名字，如 M.O.N.D.A.Y.]，全称是 Male Operational Network Digital Assistant Youth（男性运行网络数字助理青年版）。您也可以直接叫我"星期一"——虽然大家都讨厌星期一，但我保证会让您的每一天都轻松高效。

---

## 关于这个项目

本仓库是 **星期一（M.O.N.D.A.Y.）** 的项目主页、说明文档和代码。代码在 [`agent/`](agent/)，最新进展（短板清单、自己改提示词、独立做项目）见 [`agent/docs/SELF_GROWTH.md`](agent/docs/SELF_GROWTH.md)。

| 项目 | 内容 |
| --- | --- |
| 代号 | M.O.N.D.A.Y. |
| 中文名 | 星期一人工数字助理 |
| 全称 | Male Operational Network Digital Assistant Youth |
| 定位 | 男性运行网络数字助理 · 青年版 |
| 状态 | 开发中：跑在 youyeetoo R1（RK3588）板子上，本地听说，会学习，会自己做项目 |

## 设计理念

- **不讨喜的名字，讨喜的体验**：星期一是一周里最不受欢迎的一天，而我要做的是相反的事。
- **少废话，多交付**：能直接给结果的事情，不做铺垫。
- **每天都在线**：周一到周日，不需要等到下一个周一。

## 能力范围（规划）

1. 日常事务编排与提醒
2. 信息检索与资料整理
3. 文档、表格、演示文稿的生成与维护
4. 代码与自动化脚本的编写、调试
5. 项目进度跟踪与复盘

## 快速开始

```bash
git clone https://github.com/Wangpengyu01/monday-digital-assistant.git
cd monday-digital-assistant
```

## 目录结构

```
monday-digital-assistant/
├── README.md              # 项目说明（本文件）
└── agent/                 # 星期一的全部代码（部署到板子的 /userdata/agent）
    ├── orchestrator/      # 主程序：听、想、说的调度
    ├── asr/ tts/ llm/     # 语音识别、语音合成（Qwen3-TTS）、本地大模型（Qwen3.5-4B + LoRA）客户端
    ├── learning/          # 学习：老师、课堂、练习、蒸馏、短板清单、考卷、自我优化、项目模式
    ├── ops/               # 快照与自救回滚
    ├── ui/                # 网页控制面板
    ├── tools/lora/        # LoRA 训练、导出、上板
    ├── systemd/           # 板子上的服务和定时任务
    ├── tests/             # 单元测试
    └── docs/              # 文档（从 SELF_GROWTH.md 看起）
```

## 路线图

- [x] 创建仓库与项目说明
- [x] 人格设定与口语化说话（本地 4B 模型 + LoRA v5）
- [x] 板子上本地听说（流式识别、Qwen3-TTS 首音约 1 秒）
- [x] 跟老师（DeepSeek）学：课堂、练习、蒸馏、每晚反思
- [x] 动手办事：只读沙箱里查板子，改东西前先问主人，危险操作先存快照
- [x] 短板清单、固定考卷、自己改提示词（考卷说变好才留下）
- [x] 项目模式：沙箱里独立完成一个目标，能上外网不能碰内网，成果自动上线、变差自动回滚
- [ ] 自己更新权重（用自己做错的题训练下一版 LoRA）

## 许可

待定。
