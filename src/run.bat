@echo off
for /l %%i in (1,1,30) do (
echo %%i
D:\ProgramData\Anaconda3\python C:\Users\18056\Desktop\paper_all\V5\neuron2seq\src\classifier.py %%i
)