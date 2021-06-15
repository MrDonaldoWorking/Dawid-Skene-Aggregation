#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <map>
#include <math.h>

double const EPS = /*CHANGE ME*/;
int const ITER = /*CHANGE ME*/;

template <typename T>
void print_matrix(std::string const& name, std::vector<std::vector<T>> const& matrix) {
    printf("%s\n", name.c_str());
    for (auto const& i : matrix) {
        for (double const j : i) {
            printf("%.6f\t", j);
        }
        printf("\n");
    }
}

inline bool gr(double const a, double const b) {
    return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * EPS);
}

inline double check_less_than_eps(double const x) {
    if (gr(x, EPS)) {
        return x;
    }
    return EPS;
}

inline double div_op(double const a, double const b) {
    double const c = a / b;
    return check_less_than_eps(c);
}

inline double mul_op(double const a, double const b) {
    double const c = a * b;
    return check_less_than_eps(c);
}

inline double sum_op(double const a, double const b) {
    double const c = a + b;
    return check_less_than_eps(c);
}

// zj[c] - вероятность того, что ответом на вопрос j ответом является c
// ew[c,k] - вероятность ошибок исполнителей
// p[c] - априорное распределение по меткам
// answers[task_id][worker_id]
void dawid_skene(std::vector<std::vector<int>> &answers, std::vector<std::vector<double>> &z) {
    // fill zeros to ew[c,k]
    std::vector<std::vector<std::vector<double>>> e(answers[0].size(), std::vector<std::vector<double>>(z[0].size(), std::vector<double>(z[0].size(), 0)));
    // calculating ew[c,k]
    for (size_t w = 0; w < answers[0].size(); ++w) { // for all performers
        // adding for all tasks
        for (size_t t = 0; t < answers.size(); ++t) { // for all tasks
            int const ans = answers[t][w];
            if (ans == -1) { // performer may produce skip
                continue;
            }
            // for this answer
            for (size_t all = 0; all < z[0].size(); ++all) { // for all answers
                e[w][all][ans] = sum_op(e[w][all][ans], z[t][all]);
            }
        }
        // normalization
        for (size_t exp = 0; exp < e[w].size(); ++exp) { // for all expected answers
            double sum = 0;
            for (size_t act = 0; act < e[w][exp].size(); ++act) { // for all actual answers
                sum = sum_op(sum, e[w][exp][act]);
            }
            for (size_t act = 0; act < e[w][exp].size(); ++act) { // for all actual answers
                e[w][exp][act] = div_op(e[w][exp][act], sum);
            }
        }
        // print_matrix("e for worker " + std::to_string(w) + ":", e[w]);
    }
    std::vector<double> prior(z[0].size(), 0);
    {
        // normalization prior
        double sum = 0;
        for (size_t label = 0; label < z[0].size(); ++label) { // for all answers
            for (size_t task = 0; task < z.size(); ++task) { // for all tasks
                prior[label] = sum_op(prior[label], z[task][label]);
                sum = sum_op(sum, z[task][label]);
            }
        }
        for (size_t label = 0; label < z[0].size(); ++label) {
            prior[label] = div_op(prior[label], sum);
        }
    }
    // create e-step matrix
    std::vector<std::vector<double>> popular(z.size(), std::vector<double>(z[0].size()));
    // copy-paste prior values to e-step matrix
    for (size_t task = 0; task < z.size(); ++task) {
        for (size_t label = 0; label < z[0].size(); ++label) {
            popular[task][label] = prior[label];
        }
    }
    // print_matrix("popular", popular);
    // make changes by multiplication
    for (size_t t = 0; t < answers.size(); ++t) { // for all tasks
        for (size_t w = 0; w < answers[0].size(); ++w) { // for all performers
            int const ans = answers[t][w];
            if (ans == -1) {
                continue;
            }
            // for this answer
            for (size_t exp = 0; exp < e[w].size(); ++exp) { // for all expected answers
                popular[t][exp] = mul_op(popular[t][exp], e[w][exp][ans]);
            }
        }
    }
    // normalization e-step matrix
    for (size_t task = 0; task < z.size(); ++task) { // for all tasks
        double sum = 0;
        for (size_t label = 0; label < z[0].size(); ++label) { // for all answers
            sum = sum_op(sum, popular[task][label]);
        }
        for (size_t label = 0; label < z[0].size(); ++label) {
            popular[task][label] = div_op(popular[task][label], sum);
        }
    }
    // replacing z after e-step
    z = popular;
    // print_matrix("z", z);
}

std::map<int, int> result(std::vector<std::vector<double>> &z) {
    std::map<int, int> res;
    for (size_t task_id = 0; task_id < z.size(); ++task_id) {
        double max_val = 0;
        int possible_ans = -1;
        for (size_t label = 0; label < z[0].size(); ++label) {
            if (gr(z[task_id][label], max_val)) {
                max_val = z[task_id][label];
                possible_ans = label;
            }
        }
        // calculating accuracy
        res[task_id] = possible_ans;
    }
    return res;
}

int main() {
    std::ifstream gin("golden_labels.tsv");
    printf("opened golden.tsv\n");
    std::map<std::string, int> task_to_num, worker_to_num, ans_to_num;
    std::map<int, std::string> num_to_task, num_to_worker, num_to_ans;
    std::map<int, int> golden;
    std::string s;
    // read tasks from golden set
    int max_ans = 0;
    while (gin >> s) {
        std::string ans;
        gin >> ans;
        if (task_to_num.find(s) == task_to_num.end()) {
            int const size = static_cast<int>(task_to_num.size());
            task_to_num[s] = size;
            num_to_task[size] = s;
            // printf("task_id: %s <=> %d\n", s.c_str(), size);
        }
        if (ans_to_num.find(ans) == ans_to_num.end()) {
            int const size = static_cast<int>(ans_to_num.size());
            ans_to_num[ans] = size;
            num_to_ans[size] = ans;
            max_ans = std::max(max_ans, size);
        }
        // saving golden set to calculate accuracy
        golden[task_to_num[s]] = ans_to_num[ans];
    }
    gin.close();
    printf("closed golden.tsv\n");

    std::ifstream cin("crowd_labels.tsv");
    // perfrom[worker_id] - list of pair (task_id, answer)
    std::vector<std::vector<std::pair<int, int>>> perform;
    std::vector<std::pair<int, int>> empty;
    int max_worker_id = 0, max_task_id = 0;
    while (cin >> s) {
        std::string g;
        std::string ans;
        cin >> g >> ans;
        // printf("line: %s\t%s\t%d\n", s.c_str(), g.c_str(), ans);
        if (task_to_num.find(g) == task_to_num.end()) {
            continue;
        }
        // s - worker, g - task, ans - answer
        if (worker_to_num.find(s) == worker_to_num.end()) {
            int const size = static_cast<int>(worker_to_num.size());
            worker_to_num[s] = size;
            num_to_worker[size] = s;
            // printf("worker_id: %s <=> %d\n", s.c_str(), size);
        }
        if (ans_to_num.find(ans) == ans_to_num.end()) {
            int const size = static_cast<int>(ans_to_num.size());
            ans_to_num[ans] = size;
            num_to_ans[size] = ans;
        }
        perform.push_back(empty);
        perform[worker_to_num[s]].push_back(std::make_pair(task_to_num[g], ans_to_num[ans]));

        max_worker_id = std::max(max_worker_id, worker_to_num[s]);
        max_task_id = std::max(max_task_id, task_to_num[g]);
        max_ans = std::max(max_ans, ans_to_num[ans]);
    }
    cin.close();
    printf("closed crowd.tsv\n");

    // initialize answers by counting
    std::vector<std::vector<int>> answers(max_task_id + 1, std::vector<int>(max_worker_id + 1, -1));
    std::vector<std::vector<double>> z(max_task_id + 1, std::vector<double>(max_ans + 1, 0));
    // printf("perform:\n");
    for (int worker_id = 0; worker_id <= max_worker_id; ++worker_id) {
        // printf("worker %d:\n", worker_id);
        for (auto const& p : perform[worker_id]) {
            // p.first - task_id, p.second - ans
            // printf("answers[%d][%d] = %d\n", p.first, worker_id, p.second);
            answers[p.first][worker_id] = p.second;
            // printf("++z[%d][%d]\n", p.first, p.second);
            ++z[p.first][p.second];
        }
    }
    printf("cleared perform\n");
    perform.clear();
    // normalization
    for (int task_id = 0; task_id <= max_task_id; ++task_id) {
        int sum = 0;
        for (int label = 0; label <= max_ans; ++label) {
            sum += static_cast<int>(z[task_id][label]);
        }
        for (int label = 0; label <= max_ans; ++label) {
            z[task_id][label] = div_op(z[task_id][label], static_cast<double>(sum));
        }
    }
    // print_matrix("answers", answers);
    // print_matrix("z initial", z);

    // dawid skene aggregation
    for (int converges_step = 0; converges_step < ITER; ++converges_step) {
        printf("=== dawid skene %d step ===\n", converges_step + 1);
        dawid_skene(answers, z);

        int cnt = 0, correct_cnt = 0;
        for (int task_id = 0; task_id <= max_task_id; ++task_id) {
            double max_val = 0;
            int possible_ans = -1;
            for (int label = 0; label <= max_ans; ++label) {
                if (gr(z[task_id][label], max_val)) {
                    max_val = z[task_id][label];
                    possible_ans = label;
                }
            }
            // calculating accuracy
            if (golden[task_id] == possible_ans) {
                ++correct_cnt;
            }
            ++cnt;
        }
        printf("Accuracy: %.5f%%\n", 100.0 * correct_cnt / cnt);
    }

    std::ofstream out("out.tsv");
    int cnt = 0, correct_cnt = 0;
    for (int task_id = 0; task_id <= max_task_id; ++task_id) {
        double max_val = 0;
        int possible_ans = -1;
        for (int label = 0; label <= max_ans; ++label) {
            if (gr(z[task_id][label], max_val)) {
                max_val = z[task_id][label];
                possible_ans = label;
            }
        }
        // calculating accuracy
        if (golden[task_id] == possible_ans) {
            ++correct_cnt;
        }
        ++cnt;
        out << num_to_task[task_id] << '\t' << num_to_ans[possible_ans] << '\n';
    }
    out.close();
    printf("Accuracy: %.5f%%", 100.0 * correct_cnt / cnt);

    return 0;
}