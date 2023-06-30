export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yliu/anaconda3/lib/
export PATH="/home/yliu/anaconda3/envs/bark_env/bin:$PATH"

MY_PATH="/project/tts/students/yining_ws/bark/bark"
export CUDA_VISIBLE_DEVICES=5
# mkdir -p /export/data1/yliu/streamlit/
cd /export/data1/yliu/streamlit/
streamlit run $MY_PATH/bin/for_webapp_demo/Home.py --server.port 8080