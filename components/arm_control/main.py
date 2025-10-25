import os
import time
import requests
import pyarrow as pa
import numpy as np
from dora import Node
from scipy.spatial.transform import Rotation as R


def post(url, route, json=None):
    """发送POST请求，修复Content-Type和URL格式问题"""
    try:
        headers = {"Content-Type": "application/json"}
        full_url = f"{url.rstrip('/')}/{route.lstrip('/')}"

        def convert_numpy_to_python(data):
            if isinstance(data, dict):
                return {k: convert_numpy_to_python(v) for k, v in data.items()}
            elif isinstance(data, (np.ndarray, list)):
                return [convert_numpy_to_python(item) for item in data]
            elif isinstance(data, (np.float32, np.float64)):
                return float(data)
            elif isinstance(data, (np.int32, np.int64)):
                return int(data)
            else:
                return data
        
        json_data = convert_numpy_to_python(json) if json else None
        resp = requests.post(full_url, json=json_data, headers=headers)
        resp.raise_for_status()  
        print(f"成功调用API: {full_url}")
        return resp
    except requests.exceptions.RequestException as e:
        print(f"API调用失败 ({full_url}): {str(e)}")
        return None


def euler2quat(ee_pose_euler: list):
    """欧拉角转四元数，确保输出为NumPy数组（后续会转为Python列表）"""
    pos_xyz = ee_pose_euler[:3]
    euler_rpy = ee_pose_euler[3:]
    rotation = R.from_euler(seq="xyz", angles=euler_rpy)
    quat_qxqyqzqw = rotation.as_quat() 
    return np.concatenate([pos_xyz, quat_qxqyqzqw])  


def main():
    arm_url = os.getenv("url", "http://127.0.0.2:5000") 
    print(f"机械臂API地址: {arm_url}")

    node = Node()

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "arm_data":
            try:
                arm_data = event["value"].to_pylist()  
                if len(arm_data) < 14:  
                    print(f"警告：arm_data长度不足（实际{len(arm_data)}，需至少14）")
                    continue

                ee_pose_euler = arm_data[-7:-1] 
                gripper = arm_data[7]
                ee_pose_quat = euler2quat(ee_pose_euler)  
                targe_pose = ee_pose_quat.copy() 
                targe_pose[2] += 1  
                print(targe_pose)
                post(arm_url, "pose", {"arr": targe_pose.tolist()})                
                gripper_pos = np.clip(int(gripper * 255), 0, 255)
                post(arm_url, "move_gripper", {"gripper_pos": gripper_pos})

            except Exception as e:
                print(f"处理arm_data错误: {str(e)}")

        elif event["type"] == "INPUT" and event["id"] == "stop":
            print("收到停止指令，停止机械臂...")
            break  # 退出事件循环

    print("Dora节点退出，清理资源...")


if __name__ == "__main__":
    main()
