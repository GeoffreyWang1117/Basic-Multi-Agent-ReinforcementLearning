#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

const int GRID_SIZE = 5;
const int NUM_ACTIONS = 4;
const double ALPHA = 0.1;
const double GAMMA = 0.95;
const double EPSILON = 0.2;
const int PATIENCE = 3;  // 提前停止的耐心参数
const double TOLERANCE = 1e-1;  // Q 值或总奖励变化的容忍度

// 环境类
class Environment {
public:
    std::pair<int, int> agent_pos;
    std::pair<int, int> goal_pos;

    Environment() {
        agent_pos = {0, 0};
        goal_pos = {GRID_SIZE - 1, GRID_SIZE - 1};
    }

    int get_state() {
        return agent_pos.first * GRID_SIZE + agent_pos.second;
    }

    bool is_done() {
        return agent_pos == goal_pos;
    }

    int step(int action) {
        switch (action) {
            case 0: agent_pos.first = std::max(agent_pos.first - 1, 0); break;
            case 1: agent_pos.first = std::min(agent_pos.first + 1, GRID_SIZE - 1); break;
            case 2: agent_pos.second = std::max(agent_pos.second - 1, 0); break;
            case 3: agent_pos.second = std::min(agent_pos.second + 1, GRID_SIZE - 1); break;
        }

        if (is_done()) {
            return 100;
        } else if (agent_pos == std::make_pair(GRID_SIZE-1, GRID_SIZE-1)) {
            return -50;
        } else {
            return -2;
        }
    }

    void reset() {
        agent_pos = {0, 0};
    }

    void render() {
        for (int i = 0; i < GRID_SIZE; ++i) {
            for (int j = 0; j < GRID_SIZE; ++j) {
                if (agent_pos == std::make_pair(i, j)) {
                    std::cout << "A ";
                } else if (goal_pos == std::make_pair(i, j)) {
                    std::cout << "G ";
                } else {
                    std::cout << "- ";
                }
            }
            std::cout << std::endl;
        }
    }
};

// Q-Learning 智能体类
class QLearningAgent {
public:
    std::vector<std::vector<double>> Q_table;

    QLearningAgent() {
        Q_table = std::vector<std::vector<double>>(GRID_SIZE * GRID_SIZE, std::vector<double>(NUM_ACTIONS, 0.0));
    }

    int choose_action(int state) {
        if ((double)rand() / RAND_MAX < EPSILON) {
            int random_action = rand() % NUM_ACTIONS;
            std::cout << "Exploring: Chose action " << random_action << " for state " << state << std::endl;
            return random_action;
        } else {
            int best_action = std::max_element(Q_table[state].begin(), Q_table[state].end()) - Q_table[state].begin();
            std::cout << "Exploiting: Chose action " << best_action << " for state " << state << std::endl;
            return best_action;
        }
    }

    double update_Q(int state, int action, int reward, int next_state) {
        double best_next_action = *std::max_element(Q_table[next_state].begin(), Q_table[next_state].end());
        double old_value = Q_table[state][action];
        Q_table[state][action] += ALPHA * (reward + GAMMA * best_next_action - Q_table[state][action]);
        std::cout << "Updated Q(" << state << "," << action << ") from " << old_value << " to " << Q_table[state][action] << std::endl;
        return std::fabs(Q_table[state][action] - old_value);
    }

    void print_Q_table() {
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
    srand(time(0));
    Environment env;
    QLearningAgent agent;

    int no_improvement_count = 0;
    double previous_total_reward = 0;

    for (int episode = 0; episode < 500; ++episode) {
        env.reset();
        int state = env.get_state();
        int total_reward = 0;
        double max_Q_change = 0;

        while (!env.is_done()) {
            int action = agent.choose_action(state);
            int reward = env.step(action);
            int next_state = env.get_state();
            double Q_change = agent.update_Q(state, action, reward, next_state);
            max_Q_change = std::max(max_Q_change, Q_change);

            state = next_state;
            total_reward += reward;
        }

        std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
        env.render();

        if (episode % 10 == 0) {
            agent.print_Q_table();
        }

        // Early stopping logic
        if (std::fabs(total_reward - previous_total_reward) < TOLERANCE && max_Q_change < TOLERANCE) {
            no_improvement_count++;
            if (no_improvement_count >= PATIENCE) {
                std::cout << "Early stopping at episode " << episode << " due to lack of improvement." << std::endl;
                break;
            }
        } else {
            no_improvement_count = 0;  // Reset if improvement is seen
        }

        previous_total_reward = total_reward;
    }

    return 0;
}
