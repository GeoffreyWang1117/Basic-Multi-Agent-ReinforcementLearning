/*
改善了懒惰情况，但没有及时停止训练，建议加入早停机制
*/
#include <iostream>  // 包含输入输出流头文件
#include <vector>    // 包含 vector 容器头文件
#include <cstdlib>   // 包含标准库函数头文件，如 rand() 和 srand()
#include <ctime>     // 包含时间相关的头文件，用于随机数种子
#include <algorithm> // 包含算法头文件，如 max_element

const int GRID_SIZE = 5;          // 定义网格的大小为 5x5
const int NUM_ACTIONS = 4;        // 定义动作的数量，上(0), 下(1), 左(2), 右(3)
const double ALPHA = 0.1;         // 定义学习率
const double GAMMA = 0.95;        // 增加折扣因子，考虑长期回报
const double EPSILON = 0.4;       // 进一步地增加 epsilon，鼓励更多探索行为

// 环境类
class Environment {
public:
    std::pair<int, int> agent_pos; // 定义智能体的当前位置，存储为 (行, 列)
    std::pair<int, int> goal_pos;  // 定义目标的位置，存储为 (行, 列)

    Environment() {  // 构造函数
        agent_pos = {0, 0};  // 初始化智能体位置为 (0, 0)
        goal_pos = {GRID_SIZE - 1, GRID_SIZE - 1};  // 初始化目标位置为 (4, 4)
    }

    int get_state() {  // 获取当前状态，状态用位置唯一编码
        return agent_pos.first * GRID_SIZE + agent_pos.second;  // 返回状态编码
    }

    bool is_done() {  // 检查是否到达目标位置
        return agent_pos == goal_pos;  // 如果智能体位置等于目标位置，则任务完成
    }

    int step(int action) {
        switch (action) {
            case 0: agent_pos.first = std::max(agent_pos.first - 1, 0); break; // 上移，确保不越界
            case 1: agent_pos.first = std::min(agent_pos.first + 1, GRID_SIZE - 1); break; // 下移，确保不越界
            case 2: agent_pos.second = std::max(agent_pos.second - 1, 0); break; // 左移，确保不越界
            case 3: agent_pos.second = std::min(agent_pos.second + 1, GRID_SIZE - 1); break; // 右移，确保不越界
        }

        if (is_done()) {
            return 100;  // 如果到达目标位置，返回奖励 100
        } else if (agent_pos == std::make_pair(GRID_SIZE-1, GRID_SIZE-1)) {
            return -50;  // 如果智能体在同一位置不动，给予惩罚
        } else {
            return -2;  // 进一步增加负奖励，鼓励智能体更积极地探索
        }
    }

    void reset() {  // 重置环境，即重置智能体的位置
        agent_pos = {0, 0};  // 将智能体位置重置为 (0, 0)
    }

    void render() {  // 渲染环境，打印出网格及智能体和目标的位置
        for (int i = 0; i < GRID_SIZE; ++i) {  // 遍历网格的每一行
            for (int j = 0; j < GRID_SIZE; ++j) {  // 遍历网格的每一列
                if (agent_pos == std::make_pair(i, j)) {  // 如果当前位置是智能体位置
                    std::cout << "A ";  // 打印 A 代表智能体
                } else if (goal_pos == std::make_pair(i, j)) {  // 如果当前位置是目标位置
                    std::cout << "G ";  // 打印 G 代表目标
                } else {
                    std::cout << "- ";  // 否则打印 - 代表空格子
                }
            }
            std::cout << std::endl;  // 每行结束后换行
        }
    }
};

// Q-Learning 智能体类
class QLearningAgent {
public:
    std::vector<std::vector<double>> Q_table;  // 定义 Q 表，存储每个状态-动作对的 Q 值

    QLearningAgent() {  // 构造函数
        Q_table = std::vector<std::vector<double>>(GRID_SIZE * GRID_SIZE, std::vector<double>(NUM_ACTIONS, 0.0));  // 初始化 Q 表，所有 Q 值为 0
    }

    int choose_action(int state) {  // 选择动作，使用 epsilon-greedy 策略
        if ((double)rand() / RAND_MAX < EPSILON) {  // 如果随机数小于 epsilon，探索
            int random_action = rand() % NUM_ACTIONS;  // 随机选择一个动作
            std::cout << "Exploring: Chose action " << random_action << " for state " << state << std::endl;
            return random_action;
        } else {  // 否则利用
            int best_action = std::max_element(Q_table[state].begin(), Q_table[state].end()) - Q_table[state].begin();  // 选择 Q 值最大的动作
            std::cout << "Exploiting: Chose action " << best_action << " for state " << state << std::endl;
            return best_action;
        }
    }

    void update_Q(int state, int action, int reward, int next_state) {  // 更新 Q 表
        double best_next_action = *std::max_element(Q_table[next_state].begin(), Q_table[next_state].end());  // 找到下一个状态中 Q 值最大的动作
        double old_value = Q_table[state][action];
        Q_table[state][action] += ALPHA * (reward + GAMMA * best_next_action - Q_table[state][action]);  // 使用 Q-Learning 更新公式更新 Q 值
        std::cout << "Updated Q(" << state << "," << action << ") from " << old_value << " to " << Q_table[state][action] << std::endl;
    }

    void print_Q_table() {  // 打印 Q 表内容
        std::cout << "Q-table:" << std::endl;
        for (size_t i = 0; i < Q_table.size(); ++i) {
            std::cout << "State " << i << ": ";
            for (size_t j = 0; j < Q_table[i].size(); ++j) {
                std::cout << Q_table[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    srand(time(0));  // 使用当前时间作为随机数种子，确保每次运行时结果不同
    Environment env;  // 创建环境对象
    QLearningAgent agent;  // 创建 Q-Learning 智能体对象

    for (int episode = 0; episode < 1000; ++episode) {  // 进行 1000 轮训练
        env.reset();  // 重置环境
        int state = env.get_state();  // 获取当前状态
        int total_reward = 0;  // 初始化总奖励为 0

        while (!env.is_done()) {  // 当任务未完成时
            int action = agent.choose_action(state);  // 智能体选择动作
            int reward = env.step(action);  // 执行动作，获得奖励
            int next_state = env.get_state();  // 获取下一个状态
            agent.update_Q(state, action, reward, next_state);  // 更新 Q 表

            state = next_state;  // 将当前状态更新为下一个状态
            total_reward += reward;  // 累加本轮的奖励
        }

        std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
        env.render();  // 渲染环境，显示智能体和目标位置

        if (episode % 10 == 0) {
            agent.print_Q_table();  // 每10个episode打印一次Q表
        }
    }

    return 0;  // 程序正常结束
}
