MMMU-Pro Dataset
================
Source: https://huggingface.co/datasets/MMMU/MMMU_Pro
Downloaded: 2026-04-21

Overview
--------
MMMU-Pro (Massive Multitask Multimodal Understanding - Pro) is a challenging
multimodal benchmark across 30 academic subjects. All questions are multiple
choice. Images can include charts, diagrams, tables, and scientific figures.

Configs
-------
Three configurations are available:
  - standard (10 options): text question + up to 7 images, 10 answer choices (A-J)
  - standard (4 options):  text question + up to 7 images, 4 answer choices (A-D)
  - vision:                question text embedded inside a single image, 4 choices

Note: "vision" config requires multimodal (vision) LLM capability. Text-only
models (qwen2.5-coder-32b, llama-3.3-70b) cannot use this config.

Datasets in this folder
-----------------------
MMMU_Pro_standard_10/
  - Config:  standard (10 options)
  - Size:    1,730 rows (test split)
  - Answers: single letter A-J

MMMU_Pro_standard_4/
  - Config:  standard (4 options)
  - Size:    1,730 rows (test split)
  - Answers: single letter A-D

MMMU_Pro_vision/
  - Config:  vision
  - Size:    1,730 rows (test split)
  - Answers: single letter A-D
  - Note:    requires vision-capable model

MMMU_Pro_standard_10_250/
  - Config:  standard (10 options), stratified subset
  - Size:    250 rows (sampled from MMMU_Pro_standard_10)
  - Sampling: stratified by subject (30/30 covered, 7-10 per subject)
              + stratified by topic_difficulty (Easy:73, Medium:118, Hard:59)
  - Seed:    42

MMMU_Pro_standard_4_250/
  - Config:  standard (4 options), stratified subset
  - Size:    250 rows (sampled from MMMU_Pro_standard_4)
  - Sampling: stratified by subject (30/30 covered, 7-10 per subject)
              + stratified by topic_difficulty (Easy:74, Medium:116, Hard:60)
  - Seed:    42

Subjects (30 total, ~57-60 samples each in full set)
-----------------------------------------------------
Science:     Math, Physics, Chemistry, Biology
Engineering: Architecture_and_Engineering, Electronics, Mechanical_Engineering,
             Energy_and_Power, Materials, Computer_Science, Agriculture
Medicine:    Clinical_Medicine, Diagnostics_and_Laboratory_Medicine,
             Basic_Medical_Science, Pharmacy, Public_Health
Business:    Finance, Economics, Accounting, Marketing, Manage
Social:      Psychology, Sociology, Geography, History
Arts:        Art, Art_Theory, Literature, Music, Design

Evaluation
----------
All configs use single-letter multiple choice answers.
Use eval_for_multiple_choice() for scoring.
