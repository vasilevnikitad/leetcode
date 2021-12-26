#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <queue>
#include <ranges>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include <bitset>


template<typename FUN, typename...ARGS>
auto check_time(FUN&& fun, ARGS&&...args) {
    struct detect_time {
        decltype(std::chrono::high_resolution_clock::now()) start_time;
        detect_time() {
            start_time = std::chrono::high_resolution_clock::now();
        }
        ~detect_time() {
            auto const end_time{std::chrono::high_resolution_clock::now()};
            std::cout << __PRETTY_FUNCTION__ << " elapsed time " <<
                      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
        }
    } dt{};
    return fun(args...);
}

class Solution {
public:
    std::vector<int> sortedSquares(std::vector<int>& nums) {
        if (!std::size(nums))
            return {};
        std::vector<int> out(std::size(nums));
        auto it{std::rbegin(out)};

        auto it_end{std::rend(nums)};
        auto it_first{std::rbegin(nums)};
        auto it_last{std::prev(it_end)};
        for (;it_last >= it_first; it++) {
            auto const sqr_first{*it_first * *it_first};
            auto const sqr_last{*it_last * *it_last};
            if (sqr_first < sqr_last) {
                *it = sqr_last;
                it_last--;
            } else {
                *it = sqr_first;
                it_first++;
            }
        }
        return out;
    }
    template<typename It>
    constexpr static inline void myrotate(It b, It middle, It e) {
        if (b == middle || middle == e)
            return;

        auto b2{middle};
        do {
            std::iter_swap(b++, b2++);
            if (b == middle)
                middle = b2;
        } while(b2 != e);

        b2 = middle;
        while (b2 != e) {
            std::iter_swap(b++, b2++);
            if (b == middle)
                middle = b2;
            else if (b2 == e)
                b2 = middle;
        }
    }

    void rotate(std::vector<int>& nums, int k) {
        auto const b{std::rbegin(nums)};
        auto const e{std::rend(nums)};
        myrotate(b, std::next(b, k % std::size(nums)), e);
    }

    template<typename It>
    inline static void move_zeroes_impl(It begin, It end) {
        for (It it{begin}; it != end; it++) {
            if (*it != 0) {
                std::iter_swap(it, begin++);
            }
        }
    }

    inline static void moveZeroes(std::vector<int>& nums) {
        move_zeroes_impl(std::begin(nums), std::end(nums));
    }

    template<typename It, typename T>
    constexpr static It log_search(It const begin, It const end, T&& val) {
        auto b{begin};
        auto e{end};
        auto get_middle = [](auto begin, auto end) {
            auto const sz{std::distance(begin, end)};
            return std::next(begin,  sz / 2);
        };

        for (It it{get_middle(b, e)}; b != e; it = get_middle(b, e)) {
            if (*it == val)
                return it;

            if (*it < val) {
                b = std::next(it);
            } else {
                e = it;
            }
        }

        return end;
    }

    template<typename It, typename T>
    constexpr static std::pair<It, It> log_search_rotated(It const begin, It const end, T&& val) {
        auto b{begin};
        auto e{end};
        auto get_middle = [](auto begin, auto end) {
            auto const sz{std::distance(begin, end)};
            return std::next(begin,  sz / 2);
        };

        for (It it{get_middle(b, e)}; b != e; it = get_middle(b, e)) {
            if (*it == val)
                return it;

            if (*it < val) {
                b = std::next(it);
            } else {
                e = it;
            }
        }

        return end;
    }

    template<typename It, typename T>
    constexpr static It log_search_lowerbound(It const begin, It const end, T&& val) {
        auto b{begin};
        auto e{end};
        auto lb{end};
        auto get_middle = [](auto begin, auto end) {
            auto const sz{std::distance(begin, end)};
            return std::next(begin,  sz / 2);
        };

        for (It it{get_middle(b, e)}; b != e; it = get_middle(b, e)) {
            if (*it == val) {
                lb = e = it;
            } else if (*it < val) {
                b = std::next(it);
            } else {
                e = it;
            }
        }
        return lb;
    }

    template<typename It, typename T>
    constexpr static It log_search_upperbound(It const begin, It const end, T&& val) {
        auto b{begin};
        auto e{end};
        auto ub{end};
        auto get_middle = [](auto begin, auto end) {
            auto const sz{std::distance(begin, end)};
            return std::next(begin,  sz / 2);
        };

        for (It it{get_middle(b, e)}; b != e; it = get_middle(b, e)) {
            if (*it == val) {
                ub = it;
                b = std::next(it);
            } else if (*it < val) {
                b = std::next(it);
            } else {
                e = it;
            }
        }
        return ub;
    }

    static std::vector<int> twoSum(std::vector<int>& numbers, int target) {
        auto const b{std::begin(numbers)};
        for (auto it{b}, e{std::end(numbers)}; it != e; it++) {
            auto const z{log_search(std::next(it), e, target - *it)}; // FIXME: z is the name that says nothing
            if (z != e)
                return {1 + static_cast<int>(std::distance(b, it)), 1 + static_cast<int>(std::distance(b, z))};
        }

        return {};
    }


    template<typename It>
    constexpr static void myreverse(It const begin, It const end) {
        for (auto ita{begin}, itb{std::prev(end)}; ita < itb; std::iter_swap(ita++, itb--)) {}
    }

    constexpr static void reverseString(std::vector<char>& s) {
        myreverse(std::begin(s), std::end(s));
    }

    template<typename It>
    constexpr static void reverse_words(It const begin, It const end) {
        for (It word_start{begin}; word_start != end; ) {
            if (*word_start != ' ') { // FIXME: support only whitespaces
                auto word_end{std::next(word_start)}; // FIXME: we can do +2
                for (; word_end != end; word_end++) {
                    if (*word_end == ' ')
                        break;
                }
                myreverse(word_start, word_end);
                word_start = word_end; // FIXME: could be optimized. In case of word_end != end we can assign word_start = word_end + 1
            } else
                word_start++;
        }
    }

    static std::string reverseWords(std::string s) {
        reverse_words(std::begin(s), std::end(s));
        return s;
    }

    static int maxSubArray_dummy(std::vector<int>& nums) {
        if (!std::size(nums))
            return 0;
        auto out = std::numeric_limits<int>::min();

        int sum{};
        for (auto const& v: nums) {
            sum += v;

            if (out < sum)
                out = sum;

            if (sum < 0)
                sum = 0;
        }
        return out;
    }

    constexpr static int calc_fib(int n) {
        return n == 0 ? 0 : n == 1 ? 1 : calc_fib(n - 2) + calc_fib(n - 1);
    }
    constexpr static int calc_tribonacci(int n) {
        return n == 0 ? 0 : (n == 1 || n == 2)  ? 1 : calc_tribonacci(n - 3) + calc_tribonacci(n - 2) + calc_tribonacci(n - 1);
    }

    template<typename It>
    consteval static void fill_fib_container(It const begin, It const end) {
        for (It it{begin}; it != end; it++) {
            *it = calc_fib(std::distance(begin, it));
        }
    }

    template<typename T, std::size_t N>
    consteval static std::array<T, N> const get_array() {
        std::array<T, N> a;
        fill_fib_container(std::begin(a), std::end(a));
        return a;
    }

    constexpr static int fib(int n) {
        return get_array<int, 31>()[n];
    }

    template<typename It>
    static int max_sub_array(It begin, It end) {
        if (begin == end)
            return 0;

        auto out = *begin;
        auto max = out;

        for (It it{std::next(begin)}; it != end; it++) {
             max = std::max(*it, max + *it);
             out = std::max(out, max);
        }
        return out;
    }

    static int maxSubArray(std::vector<int>& nums) {
        return max_sub_array(std::begin(nums), std::end(nums));
    }

    template<typename It>
    constexpr static bool contains_duplicate(It const begin, It const end) {
        if constexpr(constexpr bool use_set{false}; use_set) {
            std::set<std::decay_t<decltype(*begin)>> keys;
            for (auto it{begin}; it < end; it++) {
                auto[unused_it, inserted]{keys.emplace(*it)};
                if (!inserted)
                    return true;
            }

            return false;
        } else {
            if (begin == end)
                return false;
            std::sort(begin, end);
            for (auto it{std::next(begin)}; it < end; it++) {
                if (*it == *std::prev(it))
                    return true;
            }
            return false;
        }
    }

    static bool containsDuplicate(std::vector<int>& nums) {
        return contains_duplicate(std::begin(nums), std::end(nums));
    }

    constexpr static int climbStairs_impl(int const n) {
        return n == 1 ? 1 : n == 2 ? 2 : climbStairs(n - 1) + climbStairs(n - 2);
    }

    static int climbStairs(int n) {
        return climbStairs_impl(n);
    }


    static int minCostClimbingStairs(std::vector<int>& cost) {
        int cost1{}, cost2{};

        for (auto const& v: cost) {
            auto current = v + std::min(cost1, cost2);
            cost2 = cost1;
            cost1 = current;
        }

        return std::min(cost1, cost2);
    }

    struct cache_init {
        template<typename It, typename T>
        constexpr cache_init(It begin, It end, T&& value) {
            std::fill(begin, end, std::forward<T>(value));
        }

        cache_init(cache_init const&) = delete;
    };

    template<typename It>
    static int length_of_longest_substring_with_map(It const begin, It const end) {
        /* static */ std::map<char, decltype(begin)> cache;

        std::size_t max_length{};
        for (auto it{begin}, it_j{it}; it != end;) {
            auto next{std::next(it)};
            auto[cache_it, inserted]{cache.emplace(*it, next)};
            if (!inserted) {
                it_j = std::max(cache_it->second, it_j);
                cache_it->second = next;
            }

            max_length = std::max(max_length, static_cast<std::size_t>(std::distance(it_j, next)));
            it = next;
        }

        return max_length;

    }
    template<typename It>
    constexpr static int length_of_longest_substring_with_array(It const begin, It const end) {
        std::decay_t<decltype(begin)> cache[std::numeric_limits<char>::max()]{};
        cache_init{std::begin(cache), std::end(cache), end};
        auto is_valid = [e = end](auto const& val) constexpr {
            return val != e;
        };

        auto access_cache = [&cache](std::size_t i) constexpr -> decltype(cache[0])& {
            return cache[i];
        };

        std::size_t max_length{};
        for (auto it{begin}, it_j{it}; it != end;) {
            auto next{std::next(it)};
            auto& cache_val{access_cache(*it)};
            if (is_valid(cache_val)) {
                it_j = std::max(cache_val, it_j);
            }
            cache_val = next;

            max_length = std::max(max_length, static_cast<std::size_t>(std::distance(it_j, next)));
            it = next;
        }

        return max_length;
    }

    inline static int lengthOfLongestSubstring(std::string const& s) {
        auto const begin{std::begin(s)};
        auto const end{std::end(s)};

        if constexpr (constexpr auto with_map{false}; with_map) {
            return length_of_longest_substring_with_map(begin, end);
        } else {
            return length_of_longest_substring_with_array(begin, end);
        }
    }

    static bool check_inclusion_sorting(std::string needle, std::string haystack) {
        auto const ndl_begin{std::begin(needle)};
        auto ndl_end{std::end(needle)};
        auto const dist_needle{std::distance(ndl_begin, ndl_end)};
        std::sort(ndl_begin, ndl_end);

        std::string tmp(std::size(needle), '\0');
        auto const hs_begin{std::begin(haystack)};
        auto const hs_end{std::end(haystack)};
        for (auto it{hs_begin}; it < hs_end; ++it) {
            if (std::distance(it, hs_end) < dist_needle)
                return false;

            std::copy_n(it, dist_needle, std::begin(tmp));
            std::sort(std::begin(tmp), std::end(tmp));

            if (std::equal(ndl_begin, ndl_end, std::begin(tmp)))
                return true;
        }

        return false;

    }

    static int search(std::vector<int>& nums, int target) {
    }

    static bool check_inclusion_window(std::string needle, std::string haystack) {
        auto const needle_sz{std::size(needle)};
        auto const haystack_sz{std::size(haystack)};
        if (needle_sz > haystack_sz)
            return false;

        constexpr std::size_t const map_sz{'z' - 'a' + 1};
        std::size_t ndl_map[map_sz]{};
        std::size_t hsk_map[map_sz]{};
        static_assert(std::size(ndl_map) == 26);
        auto get_index = [](char const ch) constexpr { return static_cast<std::size_t>(ch - 'a');};

        for (std::size_t i{}; i < needle_sz; ++i) {
            ndl_map[get_index(needle[i])]++;
            hsk_map[get_index(haystack[i])]++;
        }

        auto const ndl_map_begin{std::begin(ndl_map)};
        auto const ndl_map_end{std::end(ndl_map)};
        auto const hsk_map_begin{std::begin(hsk_map)};
        for (std::size_t i{}; i < haystack_sz - needle_sz; ++i) {
            if (std::equal(ndl_map_begin, ndl_map_end, hsk_map_begin))
                return true;
            hsk_map[get_index(haystack[i + needle_sz])]++;
            hsk_map[get_index(haystack[i])]--;
        }
        return std::equal(ndl_map_begin, ndl_map_end, hsk_map_begin);
    }

    static bool checkInclusion(std::string needle, std::string haystack) {
#if 0
        return check_inclusion_sorting(std::move(needle), std::move(haystack));
#else
        return check_inclusion_window(std::move(needle), std::move(haystack));
#endif
    }

    static bool searchMatrix(std::vector<std::vector<int>> const& matrix, int target) {
        auto const m{std::size(matrix)};
        if (!m)
            return false;
        auto const n{std::size(matrix[0])};
        auto const sz{n * m};
        auto get_middle = [](auto begin, auto end) constexpr {
            auto const sz{end - begin};
            return begin + sz / 2;
        };

        if (!sz)
            return false;

        std::size_t b{0};
        std::size_t e{sz};
        for (auto it{get_middle(b, e)}; b != e; it = get_middle(b, e)) {
            auto const value{matrix[it / n][it % n]};
            if (value == target)
                return true;

            if (value < target) {
                b = it + 1;
            } else {
                e = it;
            }
        }

        return false;
    }

    static constexpr std::size_t const sudoku_row_sz{9};
    static constexpr std::size_t const sudoku_column_sz{9};
    static constexpr std::size_t const sudoku_square_side{3};
    static_assert(sudoku_row_sz == sudoku_column_sz);
    static bool isValidSudoku(std::vector<std::vector<char>> const& board) {

        auto validate = [] (auto const& vec2d, std::size_t const column_begin, std::size_t const column_end,
                            std::size_t const roll_begin, std::size_t const roll_end) constexpr {
            auto get_index = [](char const ch) {
                return static_cast<std::size_t>(ch - '0');};
            bool cache[10]{};

            for (auto i{column_begin}; i != column_end; i++) {
                for (auto j{roll_begin}; j != roll_end; j++) {
                    if (auto const value{vec2d[i][j]}; value != '.') {
                        auto const index{get_index(value)};
                        if (cache[index])
                            return false;
                        cache[index] = true;
                    }
                }
            }

            return true;
        };

        // validate lines
        for (std::size_t i{}, next{1}; i < sudoku_column_sz; i = next, next++) {
            if (!validate(board, i, next, 0, sudoku_row_sz))
                return false;
        }

        // validate rows
        for (std::size_t i{}, next{1}; i < sudoku_row_sz; i = next, next++) {
            if (!validate(board, 0, sudoku_column_sz, i, next))
                return false;
        }

        // validate squares
        for (std::size_t i{}, i_next{sudoku_square_side}; i < sudoku_column_sz; i = i_next, i_next += sudoku_square_side) {
                for (std::size_t j{}, j_next{sudoku_square_side}; j < sudoku_column_sz; j = j_next, j_next += sudoku_square_side) {
                if (!validate(board,
                              i, i_next,
                              j, j_next))
                    return false;
            }
        }

        return true;
    }

    template<typename T>
    inline static void flood(T& image, std::size_t sr, std::size_t sc, int old_color, int new_color) {
        auto const row_sz{std::size(image[0])};
        auto const column_sz{std::size(image)};

        if (sr >= column_sz || sc >= row_sz)
            return;

        if (image[sr][sc] == old_color) {
            image[sr][sc] = new_color;
            flood(image, sr + 1,    sc,     old_color, new_color);
            flood(image, sr - 1,    sc,     old_color, new_color);
            flood(image, sr,        sc + 1, old_color, new_color);
            flood(image, sr,        sc - 1, old_color, new_color);
        }
    }

    static std::vector<std::vector<int>> floodFill(std::vector<std::vector<int>>& image, int sr, int sc, int new_color) {
        if (auto old_color{image[sr][sc]}; old_color != new_color)
            flood(image, sr, sc, old_color, new_color);
        return image;
    }

#if 0
    static std::string add_binary(std::string const a, std::string const b) {
        std::string out;
        out.reserve(std::max(std::size(a), std::size(b)) + 1);

        bool carry{};
        auto& biggest{std::size(a) < std::size(b) ? a : b};
        auto& smallest{std::size(a) >= std::size(b) ? a : b};

        for (std::size_t i{std::size(smallest)}; i < std::size(smallest); i--) {
            bool const biggest_bit(biggest[i] - '0');
            bool const smallest_bit(smallest[i] - '0');

            if (biggest_bit && smallest_bit) {
                carry = true;
            } else {
                if (biggest_bit)
                    out +=
            }

            out += carry ? '1' : '0';

            if (biggest_bit || sma
                out += carry ? '1' : '0';
                carry = false;
            } else {
                if (smallest[i])
            }
        }
    }
#endif

    template<typename T>
    inline static int area_of_island(T& grid,
                                     std::size_t sr,
                                     std::size_t sc,
                                     std::size_t const row_sz,
                                     std::size_t const column_sz) {
        enum {
            WATER = 0,
            ISLAND = 1,
            CHECKED_ISLAND = 2,
        };
        if (sr >= column_sz || sc >= row_sz || grid[sr][sc] != ISLAND)
            return 0;

        int island_sz;

        grid[sr][sc] = CHECKED_ISLAND, island_sz = 1;
        island_sz += area_of_island(grid, sr + 1, sc    , row_sz, column_sz);
        island_sz += area_of_island(grid, sr - 1, sc    , row_sz, column_sz);
        island_sz += area_of_island(grid, sr    , sc + 1, row_sz, column_sz);
        island_sz += area_of_island(grid, sr    , sc - 1, row_sz, column_sz);
        return island_sz;
    }

    static int maxAreaOfIsland(std::vector<std::vector<int>>& grid) {
        auto const column_sz{std::size(grid)};
        auto const row_sz{column_sz ? std::size(grid[0]) : 0};

        int max_island_area{};
        for (std::size_t i{}; i < column_sz; ++i) {
            for (std::size_t j{}; j < row_sz; ++j) {
                max_island_area = std::max(max_island_area, area_of_island(grid, i, j, row_sz, column_sz));
            }
        }

        return max_island_area;
    }


    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };

    static TreeNode* merge_trees(TreeNode* const root, TreeNode* const root2) {
        if (!root)
            return root2;

        if (!root2)
            return root;

        root->val += root2->val;
        root->left = merge_trees(root->left, root2->left);
        root->right = merge_trees(root->right, root2->right);
        return root;
    }

    static TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        return merge_trees(root1, root2);
    }


    class Node {
    public:
        int val;
        Node* left;
        Node* right;
        Node* next;

        Node() : val(0), left(NULL), right(NULL), next(NULL) {}

        Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

        Node(int _val, Node* _left, Node* _right, Node* _next)
                : val(_val), left(_left), right(_right), next(_next) {}
    };

    static void connect_impl(Node* const left, Node* const right) {
        Node* const right_left{right ? right->left : nullptr};
        if (right) {
            connect_impl(right_left, right->right);
            connect_impl(right->right, nullptr);
        }

        if (left) {
            left->next = right;
            connect_impl(left->left, left->right);
            connect_impl(left->right, right_left);
        }

    }

    static Node* connect(Node* root) {
        connect_impl(root, nullptr);
        return root;
    }

    struct ListNode {
        int val;
        ListNode *next;
        ListNode() : val(0), next(nullptr) {}
        ListNode(int x) : val(x), next(nullptr) {}
        ListNode(int x, ListNode *next) : val(x), next(next) {}
    };

    static ListNode* reverse_list_impl(ListNode* const head) {
        if (auto* const next{head->next}; next) {
            auto* const new_head{reverseList(next)};
            next->next = head;
            return new_head;
        }

        return head;
    }

    inline static ListNode* reverseList(ListNode* const head) {
        if (!head)
            return nullptr;

        auto* const out{reverse_list_impl(head)};
        head->next = nullptr;
        return out;
    }

    static std::vector<int> intersect(std::vector<int>& nums1, std::vector<int>& nums2) {
        //FIXME: it's not done yet!
        std::vector<int> out;
        out.reserve(std::min(std::size(nums1), std::size(nums2)));

        auto const nums1_b{std::begin(nums1)};
        auto const nums1_e{std::end(nums1)};
        auto const nums2_b{std::begin(nums2)};
        auto const nums2_e{std::end(nums2)};

        std::sort(nums1_b, nums1_e);
        std::sort(nums2_b, nums2_e);

        return out;
    }

    template<typename T, typename A, typename A2,
            template<typename, typename> class C,
            template<typename, typename> class C2>
    inline static auto update_matrix_impl(C<C2<T, A2>, A> const& mat) {
        auto const rows_count{std::size(mat)};
        if (!rows_count)
            return mat;

        auto const cols_count{std::size(mat[0])};
        std::remove_const_t<C<std::remove_const_t<C2<std::remove_const_t<T>, A2>>, A>> dist(rows_count, C2<T, A2>(cols_count, std::numeric_limits<T>::max() / 2));

        if (mat[0][0] == 0)
            dist[0][0] = 0;
        for (decltype(std::size(dist)) j{1}; j < cols_count; j++) {
            auto& square{dist[0][j]};
            if (mat[0][j] == 0)
                square = 0;
            else {
                square = std::min(square, 1 + dist[0][j - 1]);
            }
        }

        for (decltype(std::size(dist)) i{1}; i < rows_count; i++) {
            auto& square{dist[i][0]};
            if (mat[i][0] == 0)
                square = 0;
            else {
                square = std::min(square, 1 + dist[i - 1][0]);
            }
        }

        for (decltype(std::size(dist)) i{1}; i < rows_count; i++) {
            for (decltype(std::size(dist[0])) j{1}; j < cols_count; j++) {
                auto& square{dist[i][j]};
                if (mat[i][j] == 0)
                    square = 0;
                else {
                    square = std::min(square, 1 + dist[i - 1][j]);
                    square = std::min(square, 1 + dist[i][j - 1]);
                }
            }
        }

        static_assert(std::is_unsigned_v<decltype(rows_count)> &&
                      std::is_unsigned_v<decltype(cols_count)>);
        auto const last_row{rows_count - 1};
        auto const last_col{cols_count - 1};

        for (auto i{last_row}; i != std::numeric_limits<decltype(i)>::max(); i--) {
            for (auto j{last_col}; j != std::numeric_limits<decltype(j)>::max(); j--) {
                auto& square{dist[i][j]};
                if (i != last_row)
                    square = std::min(square, 1 + dist[i + 1][j]);
                if (j != last_col)
                    square = std::min(square, 1 + dist[i][j + 1]);
            }
        }

        return dist;
    }

    static std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>> const& mat) {
        return update_matrix_impl(mat);
    }

#if 0
    static int oranges_rotting_impl(std::vector<std::vector<int>>& grid) {
        int max_days{};
        auto const rows_count{std::size(grid)};
        if (!rows_count)
            return max_days;
        auto const cols_count{std::size(grid[0])};


        constexpr auto const invalid_value{std::numeric_limits<int>::max() / 2};
        std::vector<std::vector<int>> dist(rows_count, std::vector<int>(cols_count, invalid_value));

        enum {
            EMPTY_CELL = 0,
            FRESH_ORANGE = 1,
            ROTTEN_ORANGE = 2,
        };

        for (std::size_t k{}; k < rows_count; ++k) {
            max_days = 0;
            for (std::size_t i{}; i < rows_count; ++i) {
                for (std::size_t j{}; j < cols_count; ++j) {
                    auto& square{dist[i][j]};
                    if (auto const type{grid[i][j]}; type == ROTTEN_ORANGE)
                        square = 0;
                    else if (type == FRESH_ORANGE) {
                        if (i) {
                            auto const& next{dist[i - 1][j]};
                            if (grid[i - 1][j] != EMPTY_CELL)
                                square = std::min(square, 1 + next);
                        }
                        if (j) {
                            auto const& next{dist[i][j - 1]};
                            if (grid[i][j - 1] != EMPTY_CELL)
                                square = std::min(square, 1 + next);
                        }
                    }
                }
            }

            auto const last_row{rows_count - 1};
            auto const last_col{cols_count - 1};
            for (std::size_t i{last_row}; i != std::numeric_limits<std::size_t>::max(); --i) {
                for (std::size_t j{last_col}; j != std::numeric_limits<std::size_t>::max(); --j) {
                    auto& square{dist[i][j]};
                    if (auto const type{grid[i][j]}; type == FRESH_ORANGE) {
                        if (i != last_row) {
                            auto const& next{dist[i + 1][j]};
                            if (grid[i + 1][j] != EMPTY_CELL)
                                square = std::min(square, 1 + next);
                        }
                        if (j != last_col) {
                            auto const& next{dist[i][j + 1]};
                            if (grid[i][j + 1] != EMPTY_CELL)
                                square = std::min(square, 1 + next);
                        }
                        max_days = std::max(max_days, square);
                    }
                }
            }
        }

        return static_cast<std::size_t>(max_days) >= rows_count * cols_count ? -1 : max_days;
    }

#else

#if 0
    static int oranges_rotting_impl(std::vector<std::vector<int>>& grid) {

        auto const dfs = [](auto& grid,
                            auto& dist,
                            std::size_t const rows_count,
                            std::size_t const cols_count,
                            std::size_t const i,
                            std::size_t const j) {
            enum {
                EMPTY_CELL = 0,
                FRESH_ORANGE = 1,
                ROTTEN_ORANGE = 2,
            };
            if (i >= rows_count || j >= cols_count)
                return 0;

            switch(grid[i][j]) {
                case EMPTY_CELL:

            }
            if (grid[i][j] == ROTTEN_ORANGE)
                return 0;

            if (grid[i])
        }
    };

#endif
#endif

    static int orangesRotting(std::vector<std::vector<int>>& grid[[maybe_unused]]) {
#if 0
        return oranges_rotting_impl(grid);
#else
        return {};
#endif
    }

    static std::vector<std::vector<int>> generate(int const numRows) {
        std::size_t const rows_count = numRows;
        std::vector<std::vector<int>> out;
        out.reserve(rows_count);
        for (std::size_t i{}; i < rows_count; ++i) {
            std::size_t const row_sz{i + 1};
            std::vector<int> row(row_sz, 1);
            for (std::size_t j{1}; j < i; j++) {
                auto const& out_b{out.back()};
                row[j] = out_b[j - 1] + out_b[j];
            }
            out.emplace_back(std::move(row));
        }

        return out;
    }
};

TEST(Pascal_s_Triangle_118, main) {
    using vec_vec_int = std::vector<std::vector<int>>;
    vec_vec_int const expected{
                {1},
               {1,1},
              {1,2,1},
             {1,3,3,1},
            {1,4,6,4,1}
    };
    EXPECT_EQ(Solution::generate(5), expected);

}

TEST(Pascal_s_Triangle_118, een) {
    using vec_vec_int = std::vector<std::vector<int>>;
    vec_vec_int const expected{
            {1},
    };
    EXPECT_EQ(Solution::generate(1), expected);

}

TEST(Rotting_oranges_994, third) {
    using vec_vec_int = std::vector<std::vector<int>>;
    vec_vec_int input{
            {2,0,1,1,1,1,1,1,1,1},
            {1,0,1,0,0,0,0,0,0,1},
            {1,0,1,0,1,1,1,1,0,1},
            {1,0,1,0,1,0,0,1,0,1},
            {1,0,1,0,1,0,0,1,0,1},
            {1,0,1,0,1,1,0,1,0,1},
            {1,0,1,0,0,0,0,1,0,1},
            {1,0,1,1,1,1,1,1,0,1},
            {1,0,0,0,0,0,0,0,0,1},
            {1,1,1,1,1,1,1,1,1,1}
    };
    EXPECT_EQ(Solution::orangesRotting(input), 58);
}

TEST(Rotting_oranges_994, second) {
    using vec_vec_int = std::vector<std::vector<int>>;
    vec_vec_int input{
            {1, 2},
    };
    EXPECT_EQ(Solution::orangesRotting(input), 1);
}

TEST(Rotting_oranges_994, main) {
    using vec_vec_int = std::vector<std::vector<int>>;
    vec_vec_int input{
            {2, 1, 1},
            {1, 1, 0},
            {0, 1, 1}
    };
    EXPECT_EQ(Solution::orangesRotting(input), 4);
}

TEST(_01_Matrix_542, main) {
    using vec_vec_int = std::vector<std::vector<int>>;

    vec_vec_int input{
            {0, 0, 0},
            {0, 1, 0},
            {1, 1, 1}
    };
    vec_vec_int const expected {
            {0, 0, 0},
            {0, 1, 0},
            {1, 2, 1}
    };

    EXPECT_EQ(Solution::updateMatrix(input), expected);
}

TEST(Reverse_Linked_List_206, main) {
    Solution::ListNode a{1}, b{2}, c{3};

    a.next = &b, b.next = &c;
    auto* result{Solution::reverseList(&a)};
    EXPECT_EQ(result, &c);
    EXPECT_EQ(result->next, &b);
    EXPECT_EQ(result->next->next, &a);
    EXPECT_EQ(result->next->next->next, nullptr);
}

TEST(Populating_Next_Right_Pointers_in_Each_Node_116, main) {
    // lazy to write test
    //Solution::connect
}

TEST(Merge_Two_Binary_Trees_617, main) {
    // lazy to write test
}

TEST(Max_Area_of_Island_695, main) {
    std::vector<std::vector<int>> grid{
            {0,0,1,0,0,0,0,1,0,0,0,0,0},
            {0,0,0,0,0,0,0,1,1,1,0,0,0},
            {0,1,1,0,1,0,0,0,0,0,0,0,0},
            {0,1,0,0,1,1,0,0,1,0,1,0,0},
            {0,1,0,0,1,1,0,0,1,1,1,0,0},
            {0,0,0,0,0,0,0,0,0,0,1,0,0},
            {0,0,0,0,0,0,0,1,1,1,0,0,0},
            {0,0,0,0,0,0,0,1,1,0,0,0,0}
    };
    EXPECT_EQ(Solution::maxAreaOfIsland(grid), 6);
}

TEST(Flood_Fill_733, main) {
    std::vector<std::vector<int>> input, expected;
#define SUPA_MACRO(x, c, y) EXPECT_EQ(Solution::floodFill(x), y);
#if 1
    EXPECT_EQ(Solution::floodFill(input = {{3,1,1}
                                               ,{1,1,0}
                                               ,{1,0,1}}, 1, 1, 2),
                                       (expected = {{3,2,2}
                                                        ,{2,2,0}
                                                        ,{2,0,1}}));


    EXPECT_EQ(Solution::floodFill(input = {{0,0,0}
                                               ,{0,1,1}}, 1, 1, 1),
              (expected = {{0,0,0}
                               ,{0,1,1}}));

#endif
    EXPECT_EQ(Solution::floodFill(input ={{0,0,0},{0,1,0}}, 0, 0, 2),
              (expected = {{2,2,2},{2,1,2}}));
}

TEST(Valid_Sudoku_36, main) {

    EXPECT_EQ(Solution::isValidSudoku({{'5','3','.','.','7','.','.','.','.'}
                                              ,{'6','.','.','1','9','5','.','.','.'}
                                           ,{'.','9','8','.','.','.','.','6','.'}
                                           ,{'8','.','.','.','6','.','.','.','3'}
                                           ,{'4','.','.','8','.','3','.','.','1'}
                                           ,{'7','.','.','.','2','.','.','.','6'}
                                           ,{'.','6','.','.','.','.','2','8','.'}
                                           ,{'.','.','.','4','1','9','.','.','5'}
                                           ,{'.','.','.','.','8','.','.','7','9'}}), true);
    EXPECT_EQ(Solution::isValidSudoku({{'5','3','.','.','7','.','.','.','.'}
                                      ,{'5','.','.','1','9','5','.','.','.'}
                                      ,{'.','9','8','.','.','.','.','6','.'}
                                      ,{'8','.','.','.','6','.','.','.','3'}
                                      ,{'4','.','.','8','.','3','.','.','1'}
                                      ,{'7','.','.','.','2','.','.','.','6'}
                                      ,{'.','6','.','.','.','.','2','8','.'}
                                      ,{'.','.','.','4','1','9','.','.','5'}
                                      ,{'.','.','.','.','8','.','.','7','9'}}), false);
    EXPECT_EQ(Solution::isValidSudoku({{'5','5','.','.','7','.','.','.','.'}
                                      ,{'2','.','.','1','9','5','.','.','.'}
                                      ,{'.','9','8','.','.','.','.','6','.'}
                                      ,{'8','.','.','.','6','.','.','.','3'}
                                      ,{'4','.','.','8','.','3','.','.','1'}
                                      ,{'7','.','.','.','2','.','.','.','6'}
                                      ,{'.','6','.','.','.','.','2','8','.'}
                                      ,{'.','.','.','4','1','9','.','.','5'}
                                      ,{'.','.','.','.','8','.','.','7','9'}}), false);
    EXPECT_EQ(Solution::isValidSudoku({{'5','5','.','.','7','.','.','.','.'}
                                      ,{'2','.','.','1','9','5','.','.','.'}
                                      ,{'.','9','8','.','.','.','.','6','.'}
                                      ,{'8','.','.','.','6','.','.','.','3'}
                                      ,{'4','.','.','8','.','3','.','.','1'}
                                      ,{'7','.','.','.','2','.','.','.','6'}
                                      ,{'.','6','.','.','.','.','2','8','.'}
                                      ,{'.','.','.','4','1','9','.','.','5'}
                                      ,{'9','.','.','.','8','.','.','7','9'}}), false);
    EXPECT_EQ(Solution::isValidSudoku({{'5','5','.','.','7','.','.','.','.'}
                                      ,{'2','.','.','1','9','5','.','.','.'}
                                      ,{'.','9','8','.','.','.','.','6','.'}
                                      ,{'8','.','.','.','6','.','.','.','3'}
                                      ,{'4','.','.','8','.','3','.','.','1'}
                                      ,{'7','.','.','.','2','.','.','.','6'}
                                      ,{'.','6','.','.','.','.','9','8','.'}
                                      ,{'.','.','.','4','1','9','.','.','5'}
                                      ,{'9','.','.','.','8','.','.','7','9'}}), false);
}

TEST(Search_a_2D_Matrix, main) {
    EXPECT_EQ(Solution::searchMatrix({{1}, {3}}, 3), true);
    EXPECT_EQ(Solution::searchMatrix({{1}}, 1), true);
    EXPECT_EQ(Solution::searchMatrix({{1, 2}}, 1), true);
    EXPECT_EQ(Solution::searchMatrix({{1, 3}}, 3), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 3), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 13), false);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 1), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 60), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 61), false);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 10), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 23), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 20), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 7), true);
    EXPECT_EQ(Solution::searchMatrix({{1,3,5,7},{10,11,16,20},{23,30,34,60}}, 0), false);
    EXPECT_EQ(Solution::searchMatrix({{},{},{}}, 0), false);
    EXPECT_EQ(Solution::searchMatrix({}, 0), false);
}

TEST(Permutation_in_String_567, main) {
    EXPECT_EQ(Solution::checkInclusion("adc", "dcda"), true);
    EXPECT_EQ(Solution::checkInclusion("a", "a"), true);
    EXPECT_EQ(Solution::checkInclusion("a", "b"), false);
    EXPECT_EQ(Solution::checkInclusion("abc", "ab"), false);
    EXPECT_EQ(Solution::checkInclusion("ab", "abc"), true);
    EXPECT_EQ(Solution::checkInclusion("abc", "baaac"), false);
    EXPECT_EQ(Solution::checkInclusion("needle", "garbageneetleneedlegarbage"), true);
    EXPECT_EQ(Solution::checkInclusion("needle", "garbageneetle_endeel_garbage"), true);
    EXPECT_EQ(Solution::checkInclusion("needle", "garbageneetle_endfeel_garbage_elteen"), false);
}

TEST(Longest_Substring_Without_Repeating_Characters_3, main) {
    EXPECT_EQ(Solution::lengthOfLongestSubstring("abcabfjk"), 6);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("au"), 2);
    EXPECT_EQ(Solution::lengthOfLongestSubstring(" "), 1);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("pwwkew"), 3);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("ajgblhalkjdsf"), 8);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("ajhgblhalkjdsfhlk"), 8);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("bbbbb"), 1);
    EXPECT_EQ(Solution::lengthOfLongestSubstring(""), 0);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("abcafghik"), 8);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("abcabcbb"), 3);
    EXPECT_EQ(Solution::lengthOfLongestSubstring("abcabcbbabcabcbb"), 3);
    //std::string const big_string{"ajhgblhalkjdsfhlkajsdhflkajsdhflkasdfjhalsdkfjhasldkjfhladfjsdfhjksdhfjjjjdhgfhgjddjjjjjjjjjjjdfhdfjdhfjdhfjjjdjfsjhfgsdjhfgjhgjdshgfjshdgfjshdgfksdjfh"};
    //EXPECT_EQ(Solution::lengthOfLongestSubstring(big_string + big_string), 8);
}

TEST(Min_Cost_Climbing_Stairs_746, main) {
    std::vector<int> input;
    EXPECT_EQ(Solution::minCostClimbingStairs(input = {10, 15, 20}), 15);
    EXPECT_EQ(Solution::minCostClimbingStairs(input = {0,2,2,1}), 2);
    EXPECT_EQ(Solution::minCostClimbingStairs(input = {}), 0);
    EXPECT_EQ(Solution::minCostClimbingStairs(input = {2}), 0);
    EXPECT_EQ(Solution::minCostClimbingStairs(input = {1,100,1,1,1,100,1,1,100,1}), 6);
}

TEST(Climbing_Stairs_70, main) {
    std::cout << "{ " << std::endl;
    for (std::size_t i{1}; i < 47; i++) {
        std::cout << Solution::climbStairs_impl(i) << ", ";
    }
    std::cout << "} " << std::endl;

    EXPECT_EQ(Solution::climbStairs(1), 1);
    EXPECT_EQ(Solution::climbStairs(2), 2);
    EXPECT_EQ(Solution::climbStairs(3), 3);
    EXPECT_EQ(Solution::climbStairs(4), 5);
    EXPECT_EQ(Solution::climbStairs(5), 8);
    EXPECT_EQ(Solution::climbStairs(8), 34);
    EXPECT_EQ(Solution::climbStairs(40), 165580141);
}

TEST(Fibonacci_number, main) {
    std::cout << "{ " << std::endl;
    for (std::size_t i{0}; i < 39; i++) {
        std::cout << Solution::calc_tribonacci(i) << ", ";
    }
    std::cout << "} " << std::endl;
    EXPECT_EQ(Solution::fib(0), 0);
    EXPECT_EQ(Solution::fib(1), 1);
    EXPECT_EQ(Solution::fib(2), 1);
    EXPECT_EQ(Solution::fib(3), 2);
    EXPECT_EQ(Solution::fib(4), 3);
    EXPECT_EQ(Solution::fib(5), 5);
    EXPECT_EQ(Solution::fib(6), 8);
    EXPECT_EQ(Solution::fib(7), 13);
    EXPECT_EQ(Solution::fib(8), 21);
}

TEST(Maximum_Subarray_53, main) {
    auto check = [](std::vector<int> v) {
        EXPECT_EQ(Solution::maxSubArray_dummy(v), Solution::maxSubArray(v));
    };

    check({-2,1,-3,4,-1,2,1,-5,4});
    check({});
    check({1});
    check({5,4,-1,7,8});
}

TEST(Contains_Duplicate_217, main) {
    std::vector<int> in;
    EXPECT_EQ(Solution::containsDuplicate(in = {}), false);
    EXPECT_EQ(Solution::containsDuplicate(in = {1}), false);
    EXPECT_EQ(Solution::containsDuplicate(in = {1, 2, 3, 4, 5}), false);
    EXPECT_EQ(Solution::containsDuplicate(in = {1, 2, 3, 4, 5, 3}), true);
    EXPECT_EQ(Solution::containsDuplicate(in = {1, 1}), true);
}

TEST(Reverse_Words_in_a_String_III_557, main) {
    auto check = [](std::string in, std::string expected) {
        EXPECT_EQ(expected, check_time(&Solution::reverseWords, in));
    };

    check("", "");
    check("Hello", "olleH");
    check("  Hello, world! ", "  ,olleH !dlrow ");
    check("Hello, world!", ",olleH !dlrow");

}

TEST(Reverse_String_344, main) {
    auto check = [](std::vector<char> in) {
        auto result{in};
        auto expected{in};
        check_time(&Solution::reverseString, result);
        check_time(std::ranges::reverse, result);

        EXPECT_EQ(expected, result);
    };

    check({});
    check({'H'});
    check({'H', 'e', 'l', 'l', 'o'});
}

TEST(two_sum_II_167, main) {
    std::vector<int> input, expected;
    EXPECT_EQ(Solution{}.twoSum(input = {1, 2, 3}, 5), (expected = {2, 3}));
    EXPECT_EQ(Solution{}.twoSum(input = {5, 10, 11, 111}, 16), (expected = {1, 3}));
    EXPECT_EQ(Solution{}.twoSum(input = {5, 10, 11, 100}, 105), (expected = {1, 4}));
    EXPECT_EQ(Solution{}.twoSum(input = {3,24,50,79,88,150,345}, 200), (expected = {3, 6}));
}

TEST(sortedSquares, main) {
    std::vector<int> a;
    std::vector<int> b;
    EXPECT_EQ(Solution{}.sortedSquares(a = { -1, 0, 1}), (b = {0, 1, 1}));
    EXPECT_EQ(Solution{}.sortedSquares(a = {}), b = {});
    EXPECT_EQ(Solution{}.sortedSquares(a = {1}), b = {1});
    EXPECT_EQ(Solution{}.sortedSquares(a = {-1}), b = {1});
    EXPECT_EQ(Solution{}.sortedSquares(a = {0}), b = {0});
    EXPECT_EQ(Solution{}.sortedSquares(a = { -5, -3, -1, 0, 1, 2, 5}), (b = {0, 1, 1, 4, 9, 25, 25}));
}


TEST(rotate, main) {
#define MEGA_VALUE 8,2,0,4,1,4,2,1,0,6,6,2,5,6,6,2,7,9,4,1,3,9,6,5,4,8,7,8,9,2,5,5,8,3,0,5,2,5,3,9,8,5,8,8,6,3,0,2,8,1,8,4,6,4,1,6,4,3,7,9,3,0,3,9,3,3,2,1,3,2,8,7,7,7,2,0,3,1,2,1,7,7,2,8,4,0,4,3,1,9,1,5,9,8,5,6,4,2,8,0,9,6,5,7,2,6,3,1,2,1,0,6,9,7,5,3,9,8,2,6,1,8,6,6,4,4,7,3,3,5,3,2,2,9,2,7,5,2,8,5,8,7,5,3,6,0,4,1,0,8,9,0,1,2,6,0,0,3,4,1,6,6,5,9,2,5,6,7,8,4,4,5,0,8,1,1,7,9,5,2,0,1,6,2,6,1,1,3,6,5,8,7,3,8,9,6,0,0,8,9,4,0,1,6,7,8,3,9,5,1,4,6,7,3,4,7,6,3,0,1,3,9,3,1,6,4,8,8,3,8,4,7,6,7,3,4,0,1,7,6,2,5,5,2,9,9,0,9,5,9,8,3,8,3,7,9,1,9,4,0,7,6,9,0,6,8,7,9,5,5,0,7,8,8,3,4,3,8,2,6,5,8,1,3,9,0,7,6,1,4,3,7,9,3,9,3,8,8,6,8,1,5,8,2,5,2,1,2,4,6,6,4,8,7,0,8,6,1,0,9,2,3,6,7,4,8,2,0,0,0,7,3,5,4,6,7,0,0,0,1,9,0,2,7,1,1,4,5,3,7,1,2,0,9,6,6,3,4,5,8,8,4,0,3,8,3,0,4,3,5,4,7,8,6,8,2,6,1,1,6,9,0,4,5,2,1,1,1,3,5,3,8,2,6,2,4,9,4,0,7,5,2,7,4,9,6,8,8,5,7,1,7,8,1,7,0,1,6,4,3,9,1,7,4,4,0,1,0,8,9,3,7,3,3,4,9,7,7,4,9,1,8,7,9,0,0,2,3,8,9,1,0,2,6,7,0,5,6,4,5,7,4,9,4,7,3,3,2,0,4,7,4,7,2,3,7,1,6,3,7,8,1,5,4,3,2,9,6,8,0,7,4,8,3,7,7,2,6,0,1,4,4,9,0,1,1,6,8,9,5,0,2,0,5,5,8,5,1,3,6,8,9,5,7,0,0,7,2,5,6,9,6,6,3,6,3,7,8,5,3,5,9,1,4,1,1,1,5,1,4,0,0,4,9,3,3,9,5,1,4,1,8,7,9,9,2,4,9,2,9,5,2,8,0,6,5,9,0,0,6,6,8,8,3,9,3,1,6,9,4,3,7,8,0,4,2,8,6,7,8,2,1,5,7,4,9,9,7,1,7,1,1,4,8,3,4,7,8,2,5,5,4,6,9,3,2,7,2,6,1,4,2,5,8,3,6,4,4,9,4,0,6,8,4,3,6,8,5,1,0,3,5,2,3,2,9,1,6,4,8,3,3,2,7,0,7,7,8,8,5,3,0,6,8,5,8,8,0,9,9,2,1,2,3,1,2,7,5,4,5,6,9,6,0,8,9,9,8,7,3,4,1,8,7,7,0,7,3,6,3,0,8,0,4,1,8,1,4,8,1,5,4,9,4,4,5,1,5,8,7,6,8,5,8,4,4,1,5,3,9,4,8,6,8,6,3,4,8,7,0,6,8,1,8,9,8,1,9,1,4,9,2,8,2,6,7,1,9,1,0,3,6,8,3,5,4,9,3,6,1,2,6,8,7,2,3,3,3,3,2,3,9,2,4,6,1,5,7,3,8,4,6,9,9,5,0,2,1,0,6,1,9,6,7,9,6,6,7,0,3,1,9,2,4,9,3,8,3,7,3,1,9,4,4,0,3,5,9,4,5,0,2,3,4,5,9,1,0,6,5,5,7,5,4,0,9,8,2,0,7,8,7,6,4,8,6,8,0,7,1,3,9,7,7,0,9,8,5,3,9,8,2,7,2,0,8,9,6,4,8,4,4,0,6,5,8,6,0,0,9,8,6,4,7,9,3,3,2,7,9,1,9,3,2,3,7,9,5,7,3,8,7,5,5,5,1,3,7,4,1,4,9,4,3,5,1,6,8,0,7,3,1,8,3,4,5,4,5,2,7,0,9,0,9,8,0,4,0,0,7,9,8,7,4,9,0,7,9,9,7,9,7,0,2,6,2,0,9,9,4,9,5,9,7,7,6,8,9,1,6,5,9,7,0,5,0,1,2,3,7,0,5,6,4,0,3,7,9,1,8,0,3,6,2,1,1,8,8,4,9,5,5,2,1,7,5,0,8,7,0,3,4,4,5,7,2,0,4,4,8,9,5,4,0,8,5,3,4,0,5,8,0,0,2,4,1,4,3,4,6,6,9,0,8,4,2,7,7,9,4,2,1,5,1,7,5,5,7,4,1,7,5,7,6,6,5,2,6,7,1,6,9,2,9,5,1,3,6,0,1,5,9,6,5,3,8,3,9,9,2,6,8,6,3,0,9,4,6,7,8,2,8,5,9,3,6,3,5,9,0,1,5,5,9,2,5,7,1,8,2,5,1,8,0,0,1,3,1,4,1,8,2,6,9,3,9,4,4,7,4,9,1,5,0,9,0,5,5,1,1,1,3,2,6,2,2,9,1,7,7,4,1,3,1,0,7,8,5,1,2,7,4,2,6,3,5,3,6,2,4,1,6,3,6,7,4,2,0,4,6,7,0,1,3,5,0,1,4,8,3,1,9,2,0,0,1,9,8,5,7,0,5,6,1,6,2,9,9,8,5,6,1,5,1,1,8,8,5,2,6,2,0,8,0,1,0,8,0,9,5,7,8,7,6,6,6,0,4,2,4,1,5,8,3,6,2,0,4,0,8,3,9,3,5,0,5,3,1,4,1,4,8,5,3,7,9,3,0,7,3,4,5,4,6,4,4,7,6,3,0,2,8,1,7,8,5,6,1,5,7,1,8,1,5,0,7,6,4,4,6,2,1,7,1,7,9,3,0,1,6,9,9,5,2,5,3,8,3,8,6,4,3,2,1,5,5,2,0,8,2,0,9,6,9,7,4,1,9,2,6,0,8,1,4,9,0,9,5,8,5,4,6,3,8,5,3,0,5,4,5,6,7,1,9,2,8,5,8,6,8,6,4,7,1,0,0,2,2,0,3,9,1,4,6,6,1,0,7,2,3,1,2,8,3,6,5,5,4,5,0,2,1,7,6,1,6,2,5,0,1,5,3,0,8,8,9,5,8,2,9,9,1,7,4,5,1,3,3,8,0,7,4,2,6,1,4,9,5,3,6,6,6,9,5,6,4,0,6,0,3,0,9,0,3,9,3,6,1,0,5,6,9,8,6,5,9,8,2,2,2,1,4,9,2,7,0,9,2,4,9,8,7,5,3,8,8,2,2,0,3,5,6,4,7,9,5,8,4,1,6,4,1,6,6,4,3,9,5,3,9,5,0,4,5,8,4,5,8,4,7,9,8,0,5,9,8,6,8,9,6,0,9,6,6,7,6,5,8,8,2,3,5,7,3,1,1,3,0,2,7,8,5,6,3,7,5,1,0,0,3,6,2,8,5,7,2,8,4,1,6,8,6,6,1,5,6,0,2,1,1,5,7,8,7,5,1,9,8,7,5,3,9,6,4,1,7,3,3,7,6,9,0,5,3,2,4,4,6,2,0,5,7,0,3,3,6,3,2,2,9,1,6,9,8,3,5,5,1,3,0,0,1,5,8,4,3,3,5,6,0,6,8,1,6,2,4,9,7,8,1,8,4,3,7,2,8,4,1,7,8,2,7,6,0,8,7,9,7,2,2,2,4,6,9,2,1,8,6,1,1,7,0,4,5,6,0,3,2,2,5,7,6,7,7,7,4,1,7,5,9,7,0,2,8,3,0,7,4,6,8,8,5,4,3,4,2,8,1,1,3,6,9,1,7,4,8,3,7,3,1,9,8,4,6,2,6,7,7,4,4,2,1,1,9,4,8,2,2,3,2,8,7,8,0,2,9,3,1,7,6,4,0,2,3,4,4,2,3,6,0,9,8,9,5,4,2,1,2,1,8,5,7,9,7,3,7,3,3,6,4,9,4,9,0,4,7,9,1,0,3,7,7,4,9,9,6,3,5,4,0,7,7,2,0,8,5,0,0,1,7,1,0,0,0,9,7,0,5,0,2,4,9,2,7,4,5,9,0,6,9,7,7,9,3,3,6,9,2,5,3,2,4,8,1,8,4,1,7,8,0,6,4,3,8,8,4,8,3,1,5,7,4,8,2,2,7,9,1,7,5,9,0,1,5,3,2,7,5,7,1,8,1,2,1,9,0,4,5,6,0,6,1,3,3,3,4,6,8,4,5,4,4,3,0,5,2,0,3,5,0,9,0,4,9,0,7,1,1,1,9,9,9,4,6,1,9,8,9,0,6,1,2,2,0,8,6,6,6,2,4,0,0,5,3,7,7,5,1,2,3,3,5,2,5,5,7,5,2,0,1,6,7,5,4,1,1,4,2,4,9,0,3,6,8,4,8,9,3,0,6,1,0,7,6,2,4,6,7,3,9,2,3,3,7,2,8,5,3,4,1,3,4,3,2,7,8,4,8,1,7,4,0,5,6,6,5,0,3,1,6,6,6,5,7,1,6,1,9,4,9,2,8,6,1,7,9,7,6,6,0,0,1,6,2,6,2,9,3,5,0,7,5,1,6,5,4,8,1,0,0,2,1,1,7,0,1,8,5,6,3,6,0,4,2,2,4,9,5,7,7,1,0,3,4,9,8,1,2,3,4,1,9,0,1,7,3,1,6,2,9,1,9,2,2,4,4,9,8,3,8,2,4,8,4,7,5,1,3,3,6,9,6,4,1,8,3,7,8,0,2,3,3,9,7,5,5,5,1,8,1,9,6,1,3,8,1,0,9,5,6,2,0,4,1,3,2,0,4,1,5,7,3,2,4,8,2,5,1,6,2,0,1,2,3,3,8,3,7,2,5,8,6,4,6,9,0,4,9,1,7,2,0,5,1,9,7,2,9,2,8,5,5,7,6,0,3,1,0,5,0,8,4,0,8,1,4,1,2,6,7,1,3,8,4,9,4,1,0,1,3,6,0,3,0,3,7,9,5,8,2,9,8,2,0,4,0,7,2,6,0,7,1,9,3,5,8,3,6,9,4,7,0,9,4,0,9,2,5,8,6,1,6,2,8,2,5,4,4,5,5,2,2,5,4,4,5,1,2,3,0,0,9,2,5,4,7,0,0,7,2,5,8,0,1,4,8,0,2,2,3,6,9,7,7,3,2,1,8,8,9,7,5,6,4,9,9,2,9,5,7,4,0,1,2,6,6,3,1,8,9,4,2,0,0,8,3,6,2,0,8,7,2,8,3,6,4,8,7,1,6,5,1,4,2,7,6,6,1,9,9,9,1,9,7,6,4,1,5,0,5,1,1,8,2,7,3,7,0,9,6,0,7,5,0,2,5,0,7,4,3,6,9,3,9,3,7,7,9,9,9,1,5,0,3,8,6,2,7,3,3,0,5,5,5,5,7,9,6,3,4,1,9,9,6,6,4,8,7,0,0,6,0,3,1,4,0,5,2,8,4,7,7,1,8,1,4,4,7,3,2,3,5,4,4,0,0,7,4,0,2,6,9,9,0,3,5,9,7,4,6,1,4,8,3,8,9,2,9,0,1,0,5,8,7,4,6,1,3,2,7,4,8,3,5,2,6,2,7,4,7,9,3,2,8,0,2,8,3,3,3,3,8,6,5,2,7,6,7,9,0,7,7,3,5,3,6,2,6,1,6,9,8,6,3,2,5,3,1,9,0,8,4,6,7,8,7,2,5,6,4,9,5,1,7,5,6,2,8,0,9,1,1,2,8,6,5,7,6,4,5,3,5,4,6,9,5,4,8,8,5,6,9,7,2,9,2,1,1,6,5,2,1,5,3,0,6,1,4,2,4,4,4,9,6,5,8,7,0,6,8,7,4,7,5,6,1,4,5,2,1,8,4,7,1,3,3,3,6,7,3,2,9,4,6,5,4,3,2,7,0,2,9,1,7,6,0,5,3,5,8,7,8,7,1,5,1,2,6,8,3,6,1,8,9,5,9,7,4,3,2,7,7,8,3,3,0,1,8,0,3,5,4,0,4,5,6,1,1,6,6,1,3,0,4,7,4,5,4,2,3,9,5,0,9,7,0,7,6,8,9,3,0,1,9,8,2,3,7,0,8,9,5,7,2,0,1,6,6,9,7,1,0,8,4,1,0,5,0,3,5,6,1,7,2,9,0,6,5,2,1,1,8,3,2,5,1,8,0,8,2,2,7,6,7,8,0,2,2,5,9,9,9,3,4,6,6,0,7,6,8,7,5,1,9,0,6,3,9,0,0,9,1,1,3,7,5,1,0,7,7,9,7,2,0,3,9,8,0,0,3,2,1,6,1,5,4,0,5,7,5,7,6,4,1,9,1,4,1,0,8,2,0,3,8,7,7,0,9,7,7,7,6,8,2,1,9,1,2,9,2,6,8,7,2,6,6,8,7,2,6,8,6,1,9,4,4,6,5,2,1,0,2,2,2,4,3,0,1,4,6,0,0,6,4,5,8,4,3,5,3,4,8,8,7,6,3,1,4,5,2,6,5,1,8,4,8,6,1,4,3,1,2,5,3,1,7,9,9,7,4,7,5,7,6,9,1,0,4,6,3,3,4,6,4,8,2,8,9,1,1,3,0,0,0,0,5,1,8,0,4,6,2,6,2,8,1,6,4,9,0,5,0,4,2,6,4,4,9,2,6,9,2,8,1,8,0,3,0,3,7,2,5,1,7,8,5,6,2,0,4,8,1,2,8,3,8,8,6,6,2,2,4,3,5,7,6,1,2,3,6,2,8,2,8,9,7,4,8,1,4,3,2,5,3,2,5,8,7,7,3,8,6,9,9,3,1,0,3,5,2,9,2,0,2,1,4,3,1,6,9,8,2,6,5,2,0,9,7,1,7,8,9,6,9,4,4,9,5,5,6,0,4,4,6,1,0,7,0,6,2,1,6,4,3,4,6,9,1,9,8,3,8,3,1,9,3,1,4,4,3,9,3,4,9,7,9,4,1,9,8,9,1,1,8,2,2,7,1,7,3,5,9,4,8,7,0,9,8,1,2,6,7,1,0,6,2,4,3,8,3,7,1,6,2,3,9,5,4,0,3,1,5,1,4,5,8,8,6,5,2,5,5,9,6,7,6,4,2,6,8,7,1,6,5,5,6,2,2,4,4,8,3,9,4,9,7,4,4,4,5,5,2,5,0,7,9,6,0,2,7,4,7,2,8,4,3,2,9,4,0,3,2,4,7,4,0,2,0,2,6,1,5,5,1,1,9,7,6,9,0,2,9,9,0,7,4,8,5,4,4,7,0,5,2,2,5,3,0,9,9,9,8,7,0,0,0,3,0,1,3,5,9,5,5,9,6,7,4,3,0,7,8,3,7,4,1,7,8,2,5,4,3,3,7,3,5,8,5,1,9,2,8,6,4,3,6,5,7,9,8,4,2,9,3,5,6,8,5,7,6,0,4,4,3,7,1,7,6,7,2,5,1,0,1,3,6,4,3,1,3,1,5,9,8,2,8,2,8,7,3,0,5,9,5,0,6,4,9,9,9,5,2,2,6,8,9,0,6,5,4,5,2,1,8,8,5,6,1,7,9,0,2,4,1,9,5,3,9,4,8,1,5,6,6,5,3,9,9,8,3,2,0,0,3,5,5,6,3,3,5,3,9,1,3,5,1,8,4,9,2,1,0,0,9,5,6,3,6,4,2,0,0,7,3,9,2,0,2,0,4,7,3,2,9,0,9,7,5,4,7,8,4,5,3,5,2,5,7,1,0,5,3,8,8,9,0,7,3,1,6,4,4,8,5,9,1,2,7,1,1,1,7,3,1,3,9,9,3,6,9,2,7,6,2,5,1,0,0,2,3,0,7,5,0,5,9,6,3,1,9,2,4,6,7,1,4,9,5,8,7,1,7,7,2,0,3,8,8,4,6,1,3,5,7,0,6,3,2,3,0,2,5,2,3,2,6,9,0,5,7,0,2,1,0,8,9,8,4,4,5,0,5,1,0,7,4,2,7,2,8,4,8,5,6,9,0,8,1,1,1,8,7,6,3,9,4,5,3,6,3,7,8,4,2,0,5,2,1,5,3,7,8,0,5,0,4,6,2,1,4,0,2,8,1,9,3,2,9,0,9,0,6,1,8,7,4,9,3,8,6,8,0,2,6,1,9,9,5,2,0,1,0,6,7,2,5,5,7,3,5,1,1,0,4,3,2,3,1,7,7,3,4,4,8,1,5,6,8,6,2,1,1,8,5,1,1,4,5,0,3,5,0,6,8,5,7,6,6,1,5,1,2,7,6,1,0,1,5,9,7,2,0,8,6,8,2,3,2,9,9,2,6,8,6,7,9,5,9,4,0,9,2,7,1,5,5,4,2,6,6,6,6,7,6,0,2,9,4,2,0,4,3,1,6,1,1,2,2,5,7,1,0,1,2,8,8,7,1,7,7,9,9,7,3,6,6,7,4,4,4,4,6,7,6,3,3,2,4,5,9,8,1,3,3,3,1,0,3,2,1,7,7,3,4,9,4,9,6,7,3,4,2,6,7,8,7,5,2,6,7,7,4,0,4,9,4,7,4,6,0,0,9,1,7,4,5,6,4,3,1,9,3,3,5,7,1,0,8,4,5,3,1,1,8,6,3,8,9,4,1,5,1,4,7,4,8,7,8,4,9,2,9,1,8,3,0,6,7,5,8,8,4,6,7,9,4,4,3,5,0,7,1,9,4,8,1,1,4,4,4,9,7,8,5,8,7,3,9,1,7,4,6,8,8,3,1,6,9,4,6,7,1,9,6,4,5,6,5,5,3,6,6,9,2,0,2,6,2,7,6,9,3,1,1,3,1,3,8,5,3,3,1,7,6,9,6,2,2,8,4,6,9,3,3,7,9,4,4,5,1,7,9,8,6,4,6,4,2,6,2,6,1,8,8,2,0,7,4,4,9,9,3,4,8,0,1,7,3,3,7,8,5,3,6,4,8,0,7,4,1,4,6,1,8,4,4,3,2,5,3,8,1,8,5,2,7,5,1,5,3,1,9,6,6,6,2,1,4,7,3,9,2,4,4,1,5,3,3,7,3,2,6,1,5,0,1,1,4,1,9,2,9,1,8,0,7,1,1,1,0,7,1,1,5,7,5,1,9,0,6,5,4,7,9,8,6,3,4,5,2,0,1,1,8,1,5,1,4,9,8,4,1,2,1,2,2,0,9,9,5,5,9,5,8,3,6,2,4,8,4,6,8,1,6,1,5,7,3,9,8,0,2,4,4,6,9,8,1,9,7,2,8,1,4,6,1,0,0,1,2,8,4,7,7,2,8,0,5,8,1,9,7,5,6,2,2,0,3,3,1,0,0,6,0,5,3,4,9,8,3,0,0,6,5,8,0,2,8,5,7,8,5,3,3,4,1,8,5,9,9,5,8,7,7,5,2,5,4,6,6,4,4,6,6,0,5,4,6,8,4,0,4,0,5,4,3,8,5,0,9,5,2,6,4,4,9,9,4,2,7,4,5,9,6,5,4,1,0,8,6,7,7,0,1,1,5,4,9,9,2,5,8,4,3,3,9,9,0,6,7,6,1,2,9,1,1,9,2,7,1,6,2,4,4,4,2,2,3,4,6,1,9,6,7,0,8,1,8,8,8,2,0,7,5,5,1,9,1,0,0,7,9,5,8,2,3,9,5,5,1,6,9,4,6,1,2,5,1,4,3,3,7,0,6,8,1,9,0,2,9,9,9,7,4,3,7,1,4,7,0,4,7,4,6,3,3,5,8,1,1,7,2,6,2,0,4,6,7,9,4,4,1,8,0,9,7,4,2,6,5,4,1,0,8,1,6,4,7,0,8,1,1,0,9,4,0,0,7,5,9,4,4,6,4,7,6,6,5,2,2,4,5,6,8,1,4,3,1,5,5,0,8,2,5,9,3,8,7,2,8,7,5,1,6,1,0,8,4,1,9,1,1,0,5,0,2,4,6,1,3,9,4,8,1,1,6,4,1,7,6,8,5,5,3,2,9,6,1,6,2,7,6,5,2,7,8,2,5,6,1,9,6,6,4,6,4,4,4,8,0,8,3,4,1,1,9,7,5,4,1,4,9,5,1,5,6,7,6,8,4,6,1,5,7,3,9,6,2,4,3,1,7,7,8,7,5,2,5,3,7,9,4,7,4,0,0,7,6,3,1,0,8,5,2,6,5,0,4,2,6,2,1,3,9,6,9,5,3,3,8,9,3,5,7,7,2,9,4,7,6,3,8,7,1,3,4,9,2,7,9,1,4,4,3,8,6,3,2,8,2,6,3,2,3,3,5,7,7,3,1,8,2,3,3,6,0,2,3,6,7,7,3,8,9,2,6,9,3,2,5,0,5,1,0,1,7,0,6,3,7,0,3,9,1,3,0,4,7,3,6,7,8,3,2,3,5,5,9,9,4,6,5,1,4,9,6,4,7,7,8,7,5,4,5,0,2,2,3,4,3,2,1,5,9,5,1,6,0,6,3,1,0,1,9,4,8,5,2,3,3,7,5,9,6,9,3,8,8,9,5,8,0,4,8,6,6,0,4,3,8,2,9,5,3,2,5,5,3,7,6,3,0,7,5,1,1,7,0,8,1,9,2,3,1,0,0,7,7,0,6,4,0,8,0,2,3,2,1,7,9,3,5,4,8,4,7,0,4,5,6,6,1,2,9,2,8,0,3,9,8,5,8,2,7,4,1,2,5,4,2,7,6,4,7,5,4,7,4,3,1,0,2,1,3,8,3,3,7,7,1,1,9,9,0,7,0,6,6,5,5,4,2,7,5,5,3,3,9,1,2,5,2,6,8,1,5,4,3,8,9,5,2,7,2,8,5,8,2,8,4,5,7,1,0,0,1,6,1,4,5,6,6,5,0,0,9,6,2,0,4,8,3,9,7,4,3,6,1,2,5,8,4,4,4,2,6,4,7,2,5,0,4,4,8,7,2,0,1,9,4,9,6,0,4,8,7,1,8,8,7,1,0,9,0,8,9,5,8,5,9,4,5,3,5,2,0,8,5,9,3,8,6,9,6,5,9,7,1,3,8,4,8,2,3,8,0,8,3,3,3,5,9,4,4,7,8,1,4,0,4,1,8,4,0,9,5,7,9,3,4,9,0,1,6,3,0,6,7,0,5,7,8,5,9,5,1,9,6,0,5,1,6,4,0,1,6,1,9,0,3,9,3,3,2,8,6,3,6,3,5,5,1,0,4,8,9,6,3,5,3,0,9,7,4,3,3,9,0,5,8,4,0,5,1,6,3,9,7,6,1,5,0,9,0,7,2,5,1,8,3,2,5,1,4,6,7,2,4,2,2,4,1,2,9,8,8,6,5,2,4,6,9,4,6,4,3,1,8,8,3,8,8,2,4,6,3,0,2,7,3,7,7,1,0,7,7,8,7,9,8,0,8,4,1,4,0,6,6,7,7,0,8,1,5,2,5,4,7,8,0,9,0,6,8,8,4,0,3,8,4,4,2,0,0,1,3,1,8,5,7,5,9,2,3,4,4,7,8,3,1,3,4,1,7,1,3,7,3,7,7,9,6,6,1,5,5,3,6,8,1,7,5,6,2,7,6,1,0,3,8,4,7,8,5,0,1,0,3,0,8,1,7,3,9,3,1,1,6,1,9,8,5,1,4,6,2,1,5,4,6,0,1,8,4,9,0,5,4,4,7,1,9,1,3,9,7,8,9,8,5,8,7,3,9,2,5,6,9,9,9,0,8,3,7,3,6,6,7,8,5,2,3,4,9,4,5,6,9,0,4,0,7,0,4,5,0,5,5,6,9,6,8,5,7,6,8,3,6,6,5,9,6,8,0,6,4,3,1,7,3,6,5,1,6,1,7,6,6,3,6,3,8,1,6,1,7,1,2,8,8,6,1,2,2,7,0,1,2,4,2,4,4,5,8,4,8,1,8,0,7,8,9,3,9,5,9,8,2,7,6,2,7,3,5,7,7,1,4,8,5,0,5,1,1,6,8,4,6,2,1,7,5,8,3,3,7,0,7,8,1,2,9,9,6,6,8,8,4,9,7,1,5,8,4,5,4,2,9,4,7,1,0,6,4,9,5,2,8,4,5,5,8,0,9,5,7,9,2,4,5,8,6,4,0,4,1,0,1,7,8,5,2,2,6,4,3,5,4,1,5,6,8,0,5,8,8,8,1,2,8,0,8,9,6,5,8,4,7,3,6,5,0,2,6,0,3,4,6,7,2,6,8,2,7,3,0,4,2,1,1,0,8,2,2,3,2,3,7,2,2,7,3,6,2,2,0,0,4,9,3,1,0,5,0,3,2,6,3,7,1,7,7,8,8,1,5,1,3,0,8,2,0,2,8,2,5,3,5,8,3,5,5,1,5,2,4,7,7,8,0,9,4,8,9,2,1,5,8,7,1,4,8,1,7,7,4,9,4,9,2,1,3,6,1,9,0,0,1,7,4,2,0,5,8,8,6,4,9,7,8,6,9,4,8,4,0,3,9,8,1,0,0,3,2,7,7,7,3,8,6,2,7,8,2,5,8,4,0,1,5,1,4,6,5,8,1,1,1,6,0,7,2,8,5,9,1,2,3,0,3,9,5,1,2,9,7,4,8,0,3,2,7,7,5,7,7,6,6,4,2,7,6,0,6,0,8,1,6,9,3,1,4,4,0,6,7,0,0,9,1,1,4,5,6,0,7,3,4,0,9,2,0,9,0,6,6,9,6,8,6,9,5,6,0,1,7,3,6,1,3,3,9,5,9,6,2,1,9,0,1,9,9,3,2,5,8,7,8,6,3,0,4,8,3,6,7,1,6,8,3,6,9,3,3,0,1,3,6,1,8,1,1,9,0,8,2,9,0,3,4,0,5,8,8,3,4,8,5,5,4,4,4,8,0,9,2,6,8,9,1,0,1,9,7,4,6,6,8,7,1,3,2,1,1,3,9,6,1,6,2,5,6,8,1,9,6,6,9,7,0,3,3,5,7,0,7,1,3,9,5,6,0,3,1,1,6,1,7,9,8,9,7,3,5,3,9,5,6,5,8,8,8,6,2,3,5,4,0,7,5,2,4,8,2,9,2,8,0,5,9,2,8,8,4,0,2,8,2,3,7,6,1,1,3,5,0,0,5,2,5,2,9,6,4,3,0,1,4,1,0,3,6,5,4,1,6,9,6,1,3,7,7,0,3,4,8,6,0,3,1,4,2,7,4,5,2,8,6,9,6,0,6,0,4,4,7,2,1,9,7,7,7,9,7,9,5,1,9,6,1,4,6,4,5,5,5,0,3,2,6,5,0,8,7,8,8,2,8,6,1,1,1,9,5,2,2,3,6,7,4,1,4,8,2,1,4,9,1,3,1,5,7,6,2,9,5,1,7,6,3,5,9,6,9,4,4,8,4,2,2,6,6,5,8,4,5,0,4,6,7,9,4,9,8,4,4,5,4,7,8,0,2,7,1,3,1,9,1,3,4,4,7,6,7,3,5,1,4,6,1,9,8,7,2,5,4,8,7,7,3,2,2,9,2,6,7,0,9,8,1,3,2,1,2,6,8,0,5,6,5,9,3,3,1,4,3,0,3,7,5,3,1,9,9,5,3,2,3,8,4,1,3,2,7,2,0,5,0,1,3,9,2,4,3,4,0,2,9,1,8,0,1,7,2,9,6,6,0,3,5,7,8,6,0,5,3,2,5,2,0,5,3,3,0,8,5,7,6,2,1,8,2,4,9,9,0,0,4,8,1,0,6,1,5,3,9,3,9,6,7,6,7,2,1,7,3,6,6,1,5,7,2,1,2,0,2,7,1,7,4,4,4,2,8,9,1,2,6,0,1,6,4,5,2,0,8,7,5,6,3,6,6,0,6,3,8,6,7,3,9,0,7,4,5,7,4,8,2,2,3,8,0,1,0,8,0,4,8,8,6,3,0,6,5,7,2,0,0,8,8,9,5,3,9,4,7,7,4,7,6,2,2,2,3,2,5,8,1,2,7,0,4,6,1,0,5,9,4,8,7,7,1,2,0,5,6,4,1,5,6,3,0,4,4,0,9,3,3,8,0,4,3,9,0,1,4,1,8,5,9,7,3,8,3,7,5,9,8,0,6,2,6,8,1,5,5,6,7,9,4,9,2,7,0,0,2,1,9,0,4,2,1,0,7,9,5,0,3,1,8,7,3,3,7,3,6,6,0,1,6,4,2,8,9,2,4,1,0,8,8,8,5,8,6,0,9,1,4,3,8,0,8,4,0,1,3,4,4,8,4,1,1,2,5,8,1,1,3,2,1,2,9,4,8,6,0,9,7,0,9,0,1,6,0,6,3,9,1,3,4,9,5,1,7,6,8,5,8,5,8,9,5,2,9,6,5,7,7,8,9,6,0,9,7,0,1,8,9,8,5,2,4,3,8,9,9,2,9,1,5,6,3,4,2,0,7,0,8,5,8,5,9,1,2,5,3,9,2,6,9,9,2,7,6,2,7,9,2,0,2,4,6,4,3,2,9,4,8,4,1,9,3,9,8,4,0,5,2,5,2,0,5,8,0,2,1,9,9,1,8,8,9,1,9,1,2,3,0,3,2,0,4,7,7,1,9,7,5,7,1,5,5,5,9,1,2,6,0,0,3,4,6,8,1,2,9,4,1,5,2,1,7,6,4,4,2,1,4,9,0,8,9,0,7,2,5,6,7,4,9,5,8,0,4,8,1,0,0,6,1,4,5,0,2,5,1,3,1,8,2,0,6,6,4,0,0,1,0,6,7,3,0,9,1,7,9,3,3,9,7,8,9,1,2,2,2,7,0,9,5,9,4,5,7,5,9,3,5,4,4,6,3,8,8,9,2,5,6,3,1,4,3,8,4,1,6,6,7,4,8,8,5,2,8,8,1,2,1,7,6,8,8,6,3,5,8,3,2,1,2,3,8,1,1,1,3,0,6,4,3,9,4,9,1,1,9,8,0,2,0,6,5,8,0,5,7,0,3,8,5,6,7,9,3,7,1,1,7,4,2,6,1,1,3,7,3,7,2,2,5,1,9,8,6,2,4,4,4,1,6,5,8,4,4,1,8,6,1,4,1,7,2,9,8,3,5,3,0,8,7,9,3,4,0,0,5,9,5,4,9,5,0,4,2,3,3,4,9,1,1,1,6,7,8,6,8,8,3,5,7,2,3,4,3,4,9,3,4,5,8,7,9,1,0,4,0,2,1,9,1,0,2,8,6,0,0,6,1,3,7,4,3,9,9,5,4,5,7,3,2,4,9,6,3,3,8,2,6,3,2,3,2,8,4,4,1,2,9,3,1,5,2,1,3,4,7,4,2,2,5,6,5,2,6,8,6,3,7,3,1,3,1,3,8,9,3,1,9,9,3,9,7,4,6,9,3,4,4,6,2,0,7,4,8,8,0,0,4,0,6,6,8,9,9,3,3,2,7,9,1,0,3,3,7,3,7,8,8,4,0,9,1,4,6,8,3,6,2,8,1,1,3,3,7,5,6,2,5,4,0,0,4,5,1,9,8,6,6,4,5,3,9,3,6,3,2,4,8,9,2,0,2,7,4,9,2,8,1,0,6,0,1,0,1,9,9,7,5,8,6,5,1,2,8,3,6,5,2,0,3,2,0,4,7,5,5,6,8,3,9,9,2,6,6,1,5,6,7,1,3,3,8,3,4,6,2,7,2,6,9,3,9,5,6,1,0,1,8,1,4,7,4,9,8,8,4,4,1,0,9,1,3,2,8,7,6,4,7,3,1,5,4,0,9,0,5,3,8,3,6,4,4,5,2,0,1,3,4,1,2,6,8,4,8,4,7,3,7,3,1,2,5,8,2,8,8,9,4,8,5,1,3,0,6,2,5,4,3,7,3,7,6,4,0,0,3,5,6,6,0,2,8,2,8,2,7,7,9,7,3,3,2,2,2,2,8,9,3,5,9,8,0,2,1,9,5,3,5,0,8,9,7,3,9,4,2,8,4,4,9,1,6,7,5,7,4,5,8,6,0,4,3,7,9,3,2,8,6,4,3,4,0,6,4,8,4,7,2,6,5,4,0,2,6,0,8,7,0,0,8,8,9,4,4,4,6,1,3,4,5,9,8,6,2,4,1,5,0,0,0,6,9,8,6,9,5,1,1,7,2,8,3,8,1,8,8,1,8,2,0,7,2,0,9,8,2,8,0,8,5,8,4,1,5,5,9,5,9,7,8,8,5,2,7,6,3,0,3,6,9,6,4,7,8,1,0,1,4,1,7,2,3,0,9,9,6,6,4,0,5,9,2,9,2,6,1,5,8,7,4,6,2,0,0,9,2,8,6,5,7,2,6,5,2,7,6,6,1,4,0,8,7,9,7,2,3,4,2,1,8,6,5,3,0,1,8,4,5,2,4,7,6,2,8,8,8,8,0,5,8,9,6,8,3,3,9,3,3,1,5,2,7,3,2,8,6,1,0,4,5,5,5,5,8,5,8,4,7,7,8,0,2,6,1,6,8,2,6,4,8,7,0,4,7,7,9,3,4,1,4,4,8,2,0,1,7,8,5,4,6,9,8,1,1,7,6,1,7,0,2,5,8,9,1,7,2,7,1,4,4,6,3,9,5,1,4,2,1,8,1,7,9,2,8,0,1,7,1,8,7,6,1,1,0,4,4,8,1,1,5,9,3,0,0,8,7,4,6,7,2,5,4,8,0,1,0,5,2,2,6,3,2,0,3,3,8,1,2,9,5,3,1,8,0,5,5,3,7,1,5,6,0,3,8,8,8,6,1,4,8,0,2,2,3,9,2,4,2,6,0,6,3,3,1,7,2,1,3,6,5,8,3,9,5,6,0,8,5,1,5,9,9,6,8,7,7,5,6,8,4,5,5,7,2,0,1,2,5,7,1,3,9,1,4,6,6,7,6,3,8,6,9,4,2,3,0,4,2,4,3,8,6,4,7,6,7,7,1,5,8,6,7,7,8,7,4,6,2,3,1,6,5,5,5,8,4,0,1,6,9,6,5,3,5,2,3,3,3,0,3,0,5,3,8,3,1,0,7,8,7,5,7,6,3,8,1,3,8,6,0,5,1,1,4,1,5,1,4,1,8,3,1,7,9,5,6,5,2,8,0,4,7,0,7,2,6,8,6,6,9,4,2,6,6,0,3,7,2,6,7,3,3,2,4,8,5,7,2,9,7,9,2,9,2,1,2,0,4,5,7,8,1,2,6,9,8,8,1,1,0,1,6,2,1,7,2,3,9,4,8,2,3,2,7,1,0,8,6,0,0,7,3,9,6,4,1,2,7,9,8,4,2,8,4,0,1,4,4,8,8,1,1,2,0,2,8,7,4,9,3,3,1,9,8,9,7,2,4,6,1,0,3,7,5,5,0,5,0,3,7,0,6,2,3,8,8,7,3,7,8,9,1,0,2,4,9,5,8,4,9,5,4,3,6,1,8,5,1,7,2,6,0,6,3,6,0,7,5,4,0,5,3,6,6,5,8,8,9,5,7,9,4,3,1,0,1,2,6,0,2,2,5,2,4,4,1,8,1,3,6,4,9,4,6,7,8,9,6,6,0,3,8,1,0,2,2,7,9,9,4,7,6,1,9,2,9,9,1,2,4,3,5,8,3,0,9,6,3,1,0,9,3,9,1,2,6,9,2,2,6,6,5,5,0,3,5,5,5,5,7,8,3,1,6,9,6,5,1,0,2,6,4,7,7,3,9,7,6,5,1,6,6,6,4,7,5,9,7,8,0,5,1,5,5,0,2,8,6,1,5,4,9,4,9,2,9,4,8,7,0,2,8,4,6,5,8,5,5,6,0,6,1,6,4,4,2,9,4,2,0,8,7,7,0,2,2,8,2,2,7,7,8,1,3,2,1,0,5,3,0,8,7,1,1,4,2,8,1,7,0,5,7,7,7,6,7,3,9,6,6,2,6,9,4,9,5,2,4,8,4,7,3,1,4,2,6,8,9,6,0,4,6,3,0,1,0,9,1,8,6,2,2,6,6,4,4,5,8,4,1,2,9,8,7,9,0,1,8,1,0,1,5,3,6,7,8,2,9,2,4,5,0,1,7,3,3,5,4,9,4,5,6,4,4,6,9,3,6,5,9,8,1,9,5,5,7,2,4,9,1,3,4,9,3,3,1,2,7,8,8,7,2,4,6,6,1,5,3,8,2,4,9,3,0,6,0,1,2,8,7,5,0,8,6,0,7,8,8,4,4,6,5,1,3,3,9,3,8,4,6,9,6,4,2,9,4,7,5,1,4,8,0,3,5,1,0,4,8,3,9,9,9,0,1,4,7,1,2,7,1,6,9,5,6,5,7,5,9,3,3,5,3,7,1,6,1,3,8,1,7,0,6,1,3,1,0,1,7,5,5,3,2,4,1,3,4,6,2,3,8,5,9,7,6,4,7,1,7,3,7,7,0,5,9,0,7,3,8,1,8,2,7,4,8,4,0,0,2,3,2,7,1,4,2,4,7,3,4,6,4,5,6,9,3,6,3,3,0,8,9,4,0,9,7,5,6,4,1,7,4,0,6,8,3,5,3,8,3,2,7,6,0,7,9,5,0,9,9,7,4,0,7,2,6,1,9,9,0,1,9,2,1,9,6,7,3,3,4,6,2,4,7,6,2,8,7,7,8,5,9,1,6,4,8,4,8,3,4,3,6,9,1,4,8,7,9,3,2,4,2,6,1,4,6,7,5,3,2,1,1,0,9,9,2,7,8,6,8,6,3,4,5,2,4,0,0,4,6,4,5,5,8,7,0,2,6,1,8,5,0,2,9,0,6,9,5,0,2,8,3,1,2,8,9,8,4,6,0,5,1,3,7,5,8,1,2,6,2,0,2,0,6,2,6,8,6,0,9,5,4,8,2,0,5,6,1,8,9,8,2,5,7,2,3,9,4,6,3,9,2,9,2,2,3,4,6,7,1,7,2,8,1,4,2,0,9,5,1,0,0,0,1,6,3,8,4,1,6,8,4,3,4,3,0,1,2,0,8,9,3,3,5,3,7,7,3,8,7,3,4,2,2,2,4,5,2,3,2,1,7,4,7,3,4,3,2,5,9,3,3,3,8,3,7,8,5,8,3,8,6,2,2,0,5,5,2,4,3,7,6,6,8,6,4,2,1,6,6,1,0,0,6,0,7,8,5,6,2,9,2,5,2,1,8,7,3,2,8,0,5,4,9,4,1,2,0,5,1,2,0,4,6,3,2,3,2,4,0,8,6,7,8,3,3,6,7,3,2,4,7,2,2,1,3,9,8,6,0,8,0,1,8,8,4,0,5,5,8,5,7,3,3,8,8,4,7,5,7,8,9,6,8,3,3,4,2,7,5,5,5,1,2,7,3,7,2,1,3,3,9,2,3,6,4,0,4,2,5,7,7,8,3,6,6,0,3,3,7,4,7,5,7,8,6,9,9,5,4,6,7,5,5,7,3,5,2,1,1,3,0,6,3,2,7,0,3,1,9,4,7,4,5,9,2,7,8,0,9,8,4,2,6,5,8,0,7,3,0,0,9,7,6,7,3,8,0,6,7,6,4,8,5,3,4,1,5,1,9,8,6,2,4,4,0,3,4,8,2,8,5,2,7,9,2,1,5,9,9,2,2,6,2,3,4,3,6,8,8,0,6,8,2,0,1,3,4,5,4,7,8,8,5,9,9,1,5,6,7,9,2,3,3,9,1,0,2,3,3,1,8,2,0,4,5,3,4,3,8,4,9,8,0,4,6,0,6,3,5,5,6,9,4,6,6,3,8,1,8,4,1,1,2,6,3,6,7,1,6,1,6,5,8,6,4,9,1,4,8,5,8,1,5,6,4,3,1,7,3,2,0,6,2,5,0,1,5,1,6,0,0,2,0,8,6,0,3,2,9,6,8,2,2,9,1,2,5,2,7,6,3,4,9,3,9,8,4,2,5,7,4,9,7,4,0,7,1,8,6,4,8,2,9,8,4,7,8,2,9,5,2,5,8,2,6,3,3,9,4,4,5,1,4,8,7,9,3,5,8,3,2,8,9,4,6,4,0,2,0,8,9,7,6,1,8,0,7,4,2,3,3,5,9,9,5,2,8,0,9,9,8,5,1,3,4,3,0,8,6,9,3,7,8,5,1,8,9,7,1,8,1,9,4,4,6,7,1,8,4,2,2,1,1,5,2,4,4,7,4,5,5,7,8,8,2,6,0,0,2,7,2,5,1,6,8,4,4,5,3,4,8,1,5,5,9,0,0,0,5,4,4,8,8,9,6,4,6,4,5,9,3,8,3,0,2,4,0,0,0,7,9,2,4,7,8,6,3,0,3,2,6,7,2,5,9,2,9,6,3,4,9,3,1,3,1,1,7,7,1,9,7,5,3,8,2,2,0,3,7,8,9,9,5,6,8,9,6,0,5,7,1,9,3,9,7,5,7,6,3,0,6,1,5,3,7,6,5,7,2,3,6,0,2,1,1,7,8,7,2,7,5,5,1,0,3,4,1,8,0,2,1,5,2,1,9,4,1,7,7,1,2,0,4,4,5,3,8,7,2,1,9,5,2,3,1,8,9,5,2,5,4,0,8,4,7,0,3,7,5,5,7,8,6,3,8,7,4,1,0,6,4,3,2,3,7,7,4,6,6,6,0,1,7,1,1,0,3,6,1,8,5,6,9,9,7,2,3,1,1,1,3,9,3,3,8,8,3,1,0,4,4,9,1,6,3,6,1,1,8,4,1,9,1,1,7,6,4,5,7,1,4,2,1,6,3,8,6,3,7,4,5,9,3,3,9,1,9,0,9,2,2,2,2,3,4,3,7,5,4,0,7,2,4,1,2,3,0,0,0,7,1,7,2,5,3,2,0,9,6,6,9,3,2,0,7,0,2,8,3,8,7,8,4,0,7,6,7,6,4,3,9,6,7,5,8,7,2,8,5,0,4,2,0,4,2,7,3,8,5,4,1,3,3,3,7,9,1,7,9,7,6,5,0,2,2,3,1,2,2,9,1,8,9,7,5,3,5,3,6,1,2,9,5,7,9,4,3,4,5,6,7,7,9,3,0,9,6,3,1,5,1,8,2,1,9,3,2,4,8,1,4,1,6,5,5,5,8,5,3,8,2,6,4,5,8,8,1,5,8,6,1,8,6,8,7,3,5,2,2,3,6,8,9,4,9,2,8,7,4,7,9,6,8,5,3,4,8,0,2,3,2,3,5,8,8,9,0,3,7,1,1,7,3,0,8,0,7,0,6,2,1,3,7,0,3,2,1,6,8,6,0,7,8,2,2,7,1,1,8,8,1,4,9,7,0,9,2,3,3,8,3,8,1,7,1,2,6,3,5,1,6,1,9,9,9,0,8,1,9,0,7,9,4,7,9,5,3,4,5,8,8,6,1,0,7,2,0,4,3,0,0,0,5,7,2,0,4,7,9,9,4,6,3,7,4,7,2,9,1,9,5,8,1,8,7,4,8,8,8,2,8,4,3,7,9,9,1,1,2,5,4,2,1,1,9,1,0,2,7,1,6,5,0,5,8,6,8,2,4,9,4,4,5,6,4,3,9,0,4,3,7,0,2,6,5,7,4,2,6,7,6,7,6,3,9,3,0,7,8,9,5,1,7,0,4,3,5,8,5,3,4,4,3,4,3,6,6,5,1,9,3,0,9,9,5,9,9,4,0,1,6,3,2,9,9,2,4,3,6,1,7,5,5,2,3,3,2,6,4,8,2,1,3,4,1,4,7,4,1,7,7,9,3,6,0,2,6,1,5,9,2,7,2,8,9,2,2,6,3,5,6,0,2,5,3,5,3,2,2,9,0,3,4,5,4,3,9,7,6,9,4,0,0,6,9,2,8,3,6,7,8,7,0,5,9,1,1,8,9,0,0,6,9,7,5,6,3,5,3,8,6,3,7,2,5,9,1,2,8,0,4,7,5,0,3,1,7,1,3,6,4,2,8,0,1,2,4,2,1,8,4,0,0,0,7,4,7,8,8,6,6,5,5,3,5,8,5,0,1,8,1,6,1,9,5,4,7,1,9,1,7,3,2,0,3,9,0,5,6,8,1,6,4,2,6,9,5,9,5,1,7,0,2,1,0,5,7,0,8,8,2,7,5,6,4,6,4,4,6,4,3,5,6,3,2,4,7,1,6,0,2,8,9,7,5,5,4,5,3,0,3,4,0,3,4,8,3,4,5,6,3,5,5,8,1,9,5,3,3,9,6,1,0,9,7,6,3,4,3,7,5,4,7,8,9,7,3,6,7,3,7,6,6,4,6,3,0,1,0,4,6,6,5,3,8,3,9,5,6,8,4,5,1,4,2,6,6,1,4,6,8,0,3,9,0,6,3,5,2,7,1,6,5,3,5,8,0,2,0,7,8,7,8,2,1,2,8,0,0,9,6,6,0,0,7,8,8,7,9,7,5,0,8,3,8,6,6,6,1,5,0,4,2,4,1,4,7,1,2,9,6,5,3,5,8,0,4,0,4,3,5,7,8,7,3,8,7,4,9,6,1,4,0,7,7,6,9,7,9,4,4,1,5,1,7,0,0,6,3,4,9,9,4,4,8,9,0,9,2,4,6,4,9,7,6,6,8,2,2,9,5,3,4,9,7,4,5,3,8,3,7,7,8,9,1,9,9,9,1,0,9,8,4,1,4,7,8,7,5,7,5,9,7,9,4,8,0,7,1,7,9,4,3,2,2,9,8,9,9,2,9,4,1,2,4,3,0,2,4,6,7,5,8,4,1,6,2,4,1,7,0,8,0,1,0,6,4,1,1,5,6,0,9,0,2,2,3,4,2,5,2,1,8,1,8,7,8,1,0,1,7,8,2,3,3,4,5,8,3,4,1,6,0,6,4,4,9,3,5,4,2,9,1,0,4,9,8,2,0,2,7,9,9,2,6,3,3,7,3,0,5,2,6,9,0,3,9,1,1,4,6,6,9,9,8,4,0,7,8,2,1,3,3,8,7,1,9,7,8,4,1,2,1,0,3,1,7,0,0,1,2,9,1,6,0,1,2,0,7,5,8,7,4,5,3,5,0,1,2,6,1,9,9,2,2,0,8,9,9,3,6,6,8,8,3,1,3,5,8,1,6,6,0,2,0,2,3,2,1,0,7,9,2,3,6,3,9,6,7,3,2,3,3,5,0,6,7,5,8,4,1,2,2,4,7,4,1,4,8,3,5,5,0,7,1,7,0,3,5,9,0,5,3,5,6,3,3,4,4,5,2,9,5,8,8,4,1,2,2,4,7,8,1,7,1,2,6,6,7,7,6,4,2,6,0,8,8,0,9,9,0,8,9,9,9,4,2,6,9,0,9,9,0,5,7,6,4,1,9,4,3,2,6,9,2,6,2,2,9,9,0,9,0,7,2,9,5,5,9,2,9,7,5,3,3,9,5,7,6,2,0,2,3,2,6,0,0,5,0,5,3,9,9,6,3,3,0,3,8,6,9,2,3,5,3,1,8,9,6,7,5,8,0,2,6,7,8,8,8,8,3,5,3,2,5,3,7,6,5,5,2,7,9,4,6,6,7,5,2,6,3,1,9,4,3,9,4,1,0,5,9,2,4,9,9,1,4,5,9,6,3,2,4,2,6,2,0,2,2,5,3,7,2,9,9,2,3,3,3,7,6,0,0,2,7,7,2,3,4,5,6,4,6,5,8,8,6,9,9,0,6,1,4,0,4,6,4,5,8,7,8,7,3,2,5,6,9,5,2,1,8,0,8,2,0,3,6,7,9,9,8,2,2,2,5,2,4,5,7,1,6,3,4,8,7,8,6,3,4,5,2,0,1,4,7,4,0,7,8,5,7,2,1,3,0,5,6,8,3,3,1,5,1,8,4,0,9,9,0,5,6,3,8,9,2,1,3,8,0,6,5,8,3,4,5,5,8,0,6,4,8,2,4,1,0,2,5,1,8,9,3,9,3,7,1,8,0,5,7,3,6,6,1,4,4,6,3,8,5,0,7,0,6,7,6,5,5,2,5,1,8,6,0,8,7,4,6,5,8,8,3,7,0,0,4,8,9,2,4,1,7,6,4,0,0,6,6,5,4,0,9,2,7,0,1,2,9,7,6,8,8,8,3,2,6,4,7,8,2,4,7,3,5,3,4,6,2,0,4,2,6,4,8,5,5,9,3,0,0,6,8,9,2,0,4,6,1,2,0,3,7,9,6,0,2,7,6,4,9,4,4,7,0,7,4,5,6,7,7,1,0,9,9,9,6,2,5,2,9,7,1,4,8,9,2,7,9,6,0,0,6,7,4,4,5,7,0,8,6,3,6,5,6,8,9,7,4,0,7,4,7,3,8,5,6,8,9,6,0,5,6,8,4,1,0,4,4,4,6,4,5,4,9,1,8,3,1,3,8,4,1,8,9,0,1,0,2,0,1,2,7,0,9,9,6,1,1,0,2,4,2,3,1,0,8,9,7,8,3,9,4,0,4,0,4,0,0,1,5,6,1,1,5,3,0,1,7,1,6,0,7,7,6,5,1,8,7,3,1,2,8,9,6,5,0,8,7,8,8,2,9,1,9,3,7,6,0,6,2,1,2,7,6,9,8,2,3,0,1,7,9,4,7,6,5,2,8,4,8,9,9,5,4,8,8,6,4,6,4,9,7,0,1,1,1,4,0,0,3,9,6,8,3,3,0,3,0,8,6,4,9,9,0,1,9,6,2,7,6,5,4,9,3,7,2,0,8,9,5,3,9,7,2,2,0,0,3,9,4,4,7,5,8,4,7,4,1,7,1,4,0,7,0,9,2,9,9,8,0,7,6,4,8,8,5,9,8,6,5,9,9,1,9,7,5,4,3,5,1,9,6,3,0,3,4,4,5,5,3,3,0,3,8,8,4,7,3,8,3,0,2,9,7,3,3,6,6,0,4,1,5,5,9,1,3,4,7,2,7,9,6,6,9,4,1,9,1,7,2,0,3,8,6,2,6,9,4,2,7,0,4,5,0,7,9,5,7,7,7,7,0,2,6,7,1,3,9,0,5,6,0,2,9,5,7,0,4,5,6,1,8,7,3,0,8,2,6,3,3,1,8,8,6,3,7,3,8,0,7,7,3,2,1,9,4,3,9,7,3,9,6,3,1,2,8,1,4,6,7,3,4,6,2,0,6,0,3,9,9,1,4,6,2,3,6,4,9,5,1,4,6,6,7,7,5,2,1,2,4,3,8,6,9,8,3,0,1,6,8,3,6,4,8,7,0,7,0,4,8,3,8,0,5,7,5,5,6,5,4,5,8,2,3,5,7,8,2,7,8,3,2,0,1,4,6,8,0,5,2,5,1,8,3,1,8,2,6,5,2,6,3,2,6,4,0,0,1,1,2,5,6,6,7,6,4,2,2,4,5,8,2,1,6,8,2,5,6,3,4,4,6,7,3,2,1,0,1,9,3,6,5,0,5,6,0,6,6,1,1,6,4,3,3,6,6,4,7,8,6,6,5,5,1,9,1,7,5,9,8,7,4,1,3,2,9,2,7,3,4,7,2,9,8,4,5,5,4,0,0,5,4,1,9,2,6,9,1,9,9,2,0,8,5,2,8,3,0,7,0,1,9,1,8,0,7,9,2,6,1,5,0,6,1,7,1,9,4,3,9,9,5,3,5,3,3,9,6,0,2,5,9,3,2,1,2,8,7,2,2,0,0,7,7,6,8,8,8,9,7,7,4,2,9,0,0,4,3,0,5,3,3,7,5,4,4,0,0,2,4,5,8,1,1,0,3,6,8,9,1,3,5,3,1,6,2,6,7,4,7,7,3,6,3,4,3,7,3,2,1,6,1,9,6,7,1,4,5,7,4,1,5,9,9,4,3,4,2,2,6,5,8,5,0,3,4,4,8,4,3,3,7,2,7,6,3,3,7,9,6,6,6,6,3,3,3,6,3,4,0,5,0,9,0,9,5,1,4,0,9,3,5,3,0,5,8,1,5,7,5,8,9,5,6,3,7,1,9,7,1,0,5,5,3,8,3,9,8,4,5,4,7,3,0,8,9,1,3,8,4,8,1,0,2,9,5,1,3,7,9,2,9,4,6,1,7,5,7,1,3,0,6,1,6,4,3,2,5,7,7,2,9,7,3,4,1,5,3,2,3,8,6,8,3,9,2,6,1,0,8,1,1,4,7,3,1,3,3,3,0,6,5,3,4,7,0,6,5,7,0,4,9,8,5,7,8,3,0,3,2,6,5,0,1,2,9,1,5,0,9,2,4,8,0,2,8,0,2,0,4,4,8,7,9,6,0,2,5,0,3,0,8,4,4,4,2,7,4,4,2,7,3,0,1,7,8,5,4,5,3,6,1,2,8,5,7,6,9,3,0,2,4,0,6,2,3,4,7,7,0,5,8,5,8,0,1,5,0,2,7,5,1,4,5,6,6,4,8,3,5,2,5,3,8,9,4,1,5,0,3,2,3,2,7,7,3,0,3,6,8,4,4,4,4,4,5,2,9,6,3,4,2,2,0,6,1,0,4,9,4,6,7,6,5,5,9,0,4,4,6,9,8,2,7,6,1,6,1,9,1,3,3,7,2,0,2,1,8,6,4,1,4,5,5,0,4,2,7,0,1,3,7,0,2,7,5,9,2,6,5,9,0,0,0,3,5,8,6,9,0,1,8,0,4,7,8,5,1,7,1,6,1,5,3,5,5,4,5,2,2,9,9,7,3,0,9,3,7,6,9,1,0,2,8,7,8,4,0,2,4,3,2,0,1,6,8,2,0,5,4,0,6,3,2,8,6,8,8,1,4,2,6,1,3,8,7,9,8,8,3,2,0,4,4,2,1,8,8,7,4,4,7,8,9,3,3,0,3,6,5,1,4,6,7,9,4,5,5,3,2,9,0,6,2,1,8,5,3,0,2,2,1,3,0,6,5,8,4,6,3,8,0,4,1,6,7,2,3,7,8,9,3,6,5,1,7,3,9,0,9,1,3,3,7,9,8,9,6,5,3,4,0,5,0,9,0,4,4,8,7,8,6,8,3,1,6,5,0,5,4,7,8,5,6,7,1,0,9,0,3,0,1,8,4,0,5,5,2,4,6,6,3,1,4,9,4,3,8,3,6,4,7,4,6,8,9,9,2,8,4,3,1,0,0,9,1,3,5,2,7,3,7,9,6,3,2,9,4,4,0,2,4,1,4,0,2,5,1,3,3,7,0,9,4,5,0,5,5,1,7,7,7,7,0,9,5,6,4,6,7,8,5,0,2,2,3,2,5,4,8,1,7,5,7,5,3,2,4,3,7,7,3,8,2,4,4,2,7,4,1,7,1,1,6,7,2,7,6,4,1,8,3,0,6,3,1,5,0,1,5,5,6,5,3,5,1,0,8,7,3,0,0,7,7,9,2,9,0,7,1,9,3,6,6,0,2,9,5,6,7,5,8,1,9,5,7,0,3,1,0,6,5,7,7,8,4,3,6,6,3,8,7,7,0,7,3,7,1,2,2,4,4,0,2,4,2,6,8,3,3,9,0,4,5,2,3,5,8,1,2,3,7,2,2,8,7,9,7,2,2,1,8,8,5,6,4,4,7,7,8,9,2,6,9,6,0,5,9,2,0,0,7,2,1,5,0,1,5,4,7,6,6,1,8,2,4,0,3,6,1,4,2,1,8,5,5,3,9,0,8,7,3,8,1,0,1,5,0,1,2,5,4,8,3,4,8,6,3,4,5,3,1,3,6,0,2,5,8,1,0,1,4,3,1,8,6,3,5,6,9,1,5,8,3,4,0,8,2,3,9,6,9,3,0,3,5,3,9,3,9,7,7,7,1,8,0,0,5,6,2,2,6,6,3,3,4,2,7,0,4,8,3,1,3,2,8,5,1,3,7,5,8,3,7,5,5,8,8,3,8,8,7,6,8,3,8,1,2,3,6,4,6,1,1,3,1,9,3,0,6,1,9,1,7,2,6,3,0,0,5,6,1,8,6,1,7,9,0,4,5,7,5,0,2,9,4,4,8,2,3,0,3,2,3,3,4,9,8,6,5,0,9,8,7,2,9,7,7,8,4,0,1,1,8,4,7,9,0,2,2,9,6,9,6,7,3,1,1,3,7,5,5,3,8,6,9,6,1,9,2,4,7,0,5,0,3,7,4,0,0,4,2,2,9,0,1,3,0,7,7,2,0,5,3,1,5,4,2,4,0,3,6,8,1,6,0,8,8,3,0,5,8,8,3,9,7,7,9,9,5,9,6,7,6,6,1,8,9,6,7,8,5,4,4,7,5,7,2,9,4,6,2,3,0,4,8,7,8,8,8,9,2,9,9,0,7,6,1,4,9,7,2,4,4,2,7,8,4,4,4,2,6,9,9,3,1,8,6,5,1,2,4,9,2,6,4,8,2,1,5,9,3,5,1,8,3,2,0,8,8,5,2,7,7,8,0,0,2,9,1,2,6,5,1,6,5,9,5,5,7,0,8,5,9,4,5,3,2,5,2,5,5,4,9,5,6,8,1,9,7,7,8,4,6,6,2,6,4,9,8,6,2,8,3,0,4,6,4,4,2,0,6,7,5,5,5,5,8,1,3,9,6,5,9,3,2,0,8,9,8,0,2,5,6,0,6,5,8,5,7,4,7,6,0,5,8,4,2,4,4,0,1,9,3,5,5,5,1,5,7,5,6,1,4,6,8,1,4,4,6,1,9,7,9,5,2,6,9,3,6,5,7,5,7,7,1,2,9,2,5,1,2,4,7,0,8,8,3,8,7,6,8,9,3,7,5,6,5,9,8,4,5,1,6,8,4,8,5,2,9,5,0,8,9,7,2,9,3,8,6,3,5,3,8,6,8,1,2,8,3,6,0,4,7,8,6,7,1,9,5,4,5,0,6,9,0,1,0,2,9,6,5,7,0,2,7,6,2,4,6,3,8,9,9,3,4,7,2,2,5,0,9,5,1,9,2,0,9,3,5,4,5,1,5,4,5,5,6,2,3,1,5,2,6,0,1,3,6,2,6,8,2,3,3,1,0,0,3,4,6,2,4,5,3,5,2,7,1,1,4,4,5,6,3,0,7,1,3,7,0,2,1,1,1,3,9,9,9,6,2,9,8,6,1,4,6,7,0,9,0,0,4,3,3,4,0,4,4,1,8,7,1,0,3,4,2,2,9,2,0,8,3,3,5,3,9,6,6,5,7,3,1,9,6,6,8,4,9,9,1,1,2,7,9,1,5,7,1,6,3,9,1,0,6,5,3,6,3,2,5,5,7,5,7,4,4,8,1,5,7,7,7,7,2,5,7,1,1,9,1,3,6,4,6,6,8,4,6,3,1,8,2,6,1,6,2,3,7,2,9,0,5,4,0,1,0,5,6,7,7,3,6,8,9,6,1,3,8,1,1,1,1,6,3,3,7,5,5,5,1,8,2,8,5,9,4,7,9,1,1,2,6,5,1,3,3,0,3,2,6,5,9,1,2,9,3,5,3,4,6,2,5,1,2,3,6,1,1,2,7,2,2,0,0,0,2,5,6,1,8,3,0,9,8,2,0,6,2,4,7,0,4,2,9,9,2,4,6,6,1,2,6,5,3,8,7,4,7,3,6,9,2,2,0,3,7,8,6,0,7,4,8,0,9,1,2,0,3,6,7,4,8,1,0,7,8,4,9,0,1,1,9,8,4,7,6,9,0,1,6,1,2,0,7,3,1,1,0,2,6,3,0,0,9,0,5,4,4,4,7,7,1,9,1,5,1,6,0,1,8,5,8,5,1,7,7,5,5,9,1,3,6,3,8,0,9,3,1,8,5,7,1,7,9,1,2,1,5,4,4,6,1,1,2,0,7,6,0,7,1,3,5,1,7,1,0,2,1,4,3,6,2,5,4,2,5,3,9,1,6,2,5,3,5,6,4,5,8,0,9,8,3,5,7,2,4,9,6,5,7,7,2,6,1,4,3,0,2,9,0,1,3,6,9,1,5,9,7,0,9,3,3,4,9,2,3,4,8,7,9,3,0,1,2,6,2,9,8,0,8,6,8,8,7,7,7,5,4,6,5,6,5,0,7,4,7,2,0,6,8,4,1,2,5,0,0,7,5,9,3,9,9,8,6,2,5,3,6,0,3,0,3,8,6,2,4,6,3,4,7,2,7,9,2,2,3,8,0,6,5,4,0,9,3,3,4,9,9,0,4,1,3,1,6,7,7,3,3,7,7,6,0,7,8,8,2,8,6,2,7,4,0,5,0,6,3,4,0,8,8,0,6,0,4,8,5,3,4,0,9,6,6,1,5,4,7,8,6,1,5,3,3,7,9,8,8,3,7,3,8,4,3,4,8,4,4,1,0,0,4,0,6,3,2,8,1,5,1,1,8,4,3,0,7,2,0,7,1,0,9,1,5,2,9,9,8,9,4,9,7,3,2,6,2,6,1,2,7,9,3,5,0,6,1,3,9,7,3,1,6,3,6,3,5,1,1,5,3,3,9,1,4,2,4,9,8,9,2,6,1,4,7,4,2,0,3,6,6,0,0,6,0,1,3,0,8,2,3,8,6,0,6,9,0,9,5,0,2,0,6,5,7,1,3,1,8,7,7,1,6,5,9,5,8,3,6,6,0,3,2,2,7,7,7,4,9,1,3,5,8,2,7,3,1,4,3,3,0,7,2,5,2,9,8,8,3,9,4,0,2,3,1,1,5,3,5,5,7,8,1,0,0,5,2,9,6,7,2,0,0,7,8,7,3,6,8,3,9,3,0,4,3,4,5,6,7,8,8,6,1,6,1,9,7,3,2,8,4,8,3,3,7,1,8,7,9,4,4,1,5,6,9,7,6,7,6,4,7,3,4,0,1,3,7,4,8,8,8,5,8,0,1,5,9,2,1,5,8,5,4,4,8,9,7,0,6,5,9,8,0,0,1,7,8,6,0,4,7,4,4,1,1,3,9,8,5,6,0,1,5,0,1,8,4,0,9,8,9,8,3,7,2,8,9,7,3,4,7,2,3,8,6,8,9,7,0,2,8,5,1,4,0,9,7,5,3,8,4,4,1,7,0,0,1,7,4,3,7,1,1,5,5,6,1,3,5,5,4,2,3,0,1,7,6,3,3,6,3,6,5,1,4,1,8,9,5,5,3,4,0,1,8,3,7,3,7,4,5,0,5,3,0,7,5,4,5,9,4,0,0,7,7,6,1,1,5,5,8,3,9,5,8,5,2,5,6,4,9,9,3,9,3,6,1,2,9,1,9,7,8,4,6,5,8,0,5,7,8,2,7,2,3,6,7,9,9,6,9,6,3,5,1,4,7,1,4,9,4,0,9,5,4,4,0,3,3,6,0,4,6,0,5,3,8,1,3,2,0,3,0,6,7,4,8,9,9,0,4,6,2,7,9,5,2,3,1,3,2,1,6,8,8,7,3,6,9,3,4,0,0,1,3,9,5,3,9,3,6,0,3,9,9,4,0,1,1,6,6,8,2,0,0,1,8,3,3,0,0,4,1,7,3,0,2,8,6,2,5,3,5,9,4,7,2,5,4,6,2,1,7,9,6,0,1,2,5,8,9,6,1,9,2,7,2,1,0,8,1,9,5,3,6,0,7,5,8,7,1,8,6,0,3,5,4,6,2,6,0,3,2,5,6,8,1,9,3,1,6,0,5,3,3,6,4,2,1,4,1,2,4,8,8,2,8,5,3,5,1,0,8,6,2,5,7,8,0,9,3,8,3,3,1,9,3,7,2,7,5,3,1,4,6,8,4,5,8,3,3,4,5,7,3,7,5,2,8,3,3,3,0,1,0,7,2,1,7,1,2,8,8,2,8,4,3,0,0,1,8,5,6,4,0,9,3,6,4,2,2,1,4,3,3,2,0,5,6,7,2,4,6,0,0,7,7,0,6,4,3,5,0,8,1,0,5,9,7,6,5,1,5,1,0,1,6,5,7,8,5,1,3,1,1,2,9,5,0,7,0,0,9,8,5,5,4,2,6,3,5,7,3,1,2,7,7,8,2,4,9,9,6,5,9,5,4,5,4,5,5,3,9,4,7,5,4,6,6,0,3,9,2,4,9,5,1,2,6,4,7,9,4,6,9,0,2,6,2,4,5,9,9,5,2,6,2,1,3,6,9,8,8,8,5,9,4,3,7,0,8,2,5,1,9,4,4,3,1,8,9,4,7,1,6,6,6,0,6,5,1,6,1,9,2,8,4,6,5,5,8,6,8,6,0,5,9,7,9,8,1,7,5,3,4,8,2,2,1,4,0,5,3,4,9,2,7,1,1,0,3,0,4,4,7,3,0,6,2,9,9,6,0,5,6,3,6,2,6,0,6,5,0,4,1,0,9,4,9,5,8,2,1,7,1,0,5,7,7,2,1,6,3,7,6,3,2,1,9,7,4,5,5,7,4,5,4,7,8,0,2,2,8,0,7,1,8,7,0,9,3,6,8,3,2,1,2,9,9,8,6,0,4,5,6,2,9,6,6,0,5,1,2,6,7,1,9,0,5,6,8,7,1,0,0,2,7,1,9,6,1,9,7,2,3,2,5,3,9,6,6,9,3,6,1,8,7,2,4,6,6,5,7,1,8,2,8,9,8,3,0,9,3,5,4,4,7,9,4,7,3,6,8,9,3,5,7,1,8,0,9,2,4,3,8,3,3,0,1,8,4,9,4,6,6,6,6,7,1,3,5,6,8,3,4,1,5,3,8,6,1,0,5,4,3,6,6,0,5,5,3,7,5,4,8,7,1,6,9,9,5,6,0,0,8,1,0,6,7,3,5,9,0,9,9,0,1,8,8,7,6,9,2,9,4,6,2,3,6,1,3,5,0,1,0,8,4,4,0,1,3,8,9,7,7,5,3,4,1,7,1,3,4,8,4,8,6,0,4,1,3,3,8,2,5,6,5,9,3,4,4,0,0,6,4,9,8,1,9,6,6,9,9,1,4,4,8,0,3,7,9,0,2,1,9,3,8,8,3,2,5,2,9,5,1,8,0,1,0,1,4,5,9,8,3,2,5,7,1,3,8,8,6,6,3,6,5,5,8,4,9,6,3,2,8,7,6,3,2,7,5,7,2,2,5,9,1,9,4,5,3,0,1,6,8,4,4,8,4,5,4,2,3,7,2,5,0,6,5,9,0,7,9,0,8,0,4,3,6,9,3,4,0,7,1,7,2,3,4,3,3,6,8,7,3,7,9,4,6,6,9,8,8,2,4,5,6,4,0,8,3,2,6,2,0,0,0,2,6,7,9,6,1,6,9,4,2,0,9,3,9,9,0,2,7,3,9,6,3,1,4,2,7,0,3,4,7,7,6,4,7,4,2,6,5,7,0,8,0,9,5,3,1,8,7,1,1,5,2,9,7,0,4,2,2,2,2,6,7,6,9,8,9,6,9,7,2,0,2,1,6,4,2,7,9,2,0,5,0,3,0,8,5,6,8,0,6,9,1,0,5,7,9,6,1,1,9,3,9,9,8,7,2,2,2,7,2,0,0,1,8,4,6,0,1,2,0,2,9,9,9,4,1,4,5,6,4,5,2,2,6,8,1,3,0,5,0,8,7,2,0,0,4,4,4,6,3,8,0,4,7,2,0,0,5,7,5,6,1,0,7,0,2,1,2,1,3,9,0,1,1,4,6,5,8,2,1,5,4,0,1,3,5,6,7,9,8,3,4,4,0,6,1,7,8,9,8,5,0,3,2,5,7,7,7,0,9,9,7,3,4,3,8,4,1,1,3,7,7,2,7,0,3,0,8,3,9,4,7,9,0,0,0,9,7,7,0,2,9,1,2,4,0,7,1,9,3,2,8,5,5,4,3,2,1,2,6,3,8,5,0,2,6,1,3,8,0,7,0,4,2,7,0,4,4,8,7,9,8,2,9,4,1,8,1,6,2,1,7,4,2,5,0,2,1,0,2,6,4,9,0,8,4,1,3,6,2,0,8,0,6,0,6,9,5,0,2,2,6,5,2,5,4,2,4,8,6,9,4,3,2,3,5,9,0,4,2,1,3,7,9,8,4,5,2,4,8,4,0,1,2,3,1,8,7,2,2,3,5,3,6,9,4,2,4,6,8,0,8,0,6,0,3,9,9,7,2,0,2,2,0,5,9,6,0,3,7,5,6,7,8,7,8,5,3,4,7,1,2,9,1,2,9,8,4,4,7,9,3,6,7,9,7,3,6,4,5,0,8,9,6,6,4,7,9,3,7,0,4,6,9,7,1,1,8,3,4,4,8,2,2,8,4,7,8,7,6,3,7,3,1,8,7,9,7,6,5,9,8,0,3,9,2,2,9,9,0,8,3,6,8,9,2,7,0,0,7,7,0,4,6,5,3,4,0,9,7,0,5,5,2,6,9,9,0,5,1,5,3,1,0,0,3,1,2,4,7,9,0,8,3,8,4,4,2,2,0,7,8,1,0,6,2,4,5,8,4,9,3,8,2,2,9,5,6,6,5,1,1,4,8,8,3,6,6,0,4,9,3,4,9,4,8,2,6,1,6,6,2,8,0,5,6,5,5,6,5,4,3,0,1,8,0,2,5,5,0,6,3,6,4,8,9,4,2,9,7,8,3,5,4,8,7,4,8,1,3,8,6,5,6,1,4,0,1,5,6,2,0,6,3,1,4,4,3,3,5,9,9,9,5,8,9,9,5,2,8,8,0,7,2,6,5,9,0,9,6,3,6,4,7,5,6,3,3,4,8,5,2,3,4,4,1,5,3,7,7,3,0,3,5,8,1,1,5,1,8,0,8,9,6,2,5,1,1,7,7,2,9,6,6,5,3,0,2,2,3,6,3,4,3,9,6,3,3,4,0,8,4,7,9,9,1,8,5,7,8,5,0,1,1,0,1,0,5,2,7,1,3,0,0,2,8,9,8,2,7,5,2,0,2,8,1,1,3,8,0,5,9,7,1,4,3,9,2,9,8,1,6,2,2,3,0,2,2,9,7,4,1,8,6,6,7,3,3,0,2,6,2,4,6,3,5,8,9,6,2,7,6,4,7,8,9,8,3,1,8,0,2,2,6,9,4,8,9,8,5,5,1,5,8,4,1,8,9,4,4,9,7,1,9,8,8,3,6,3,5,6,5,8,1,3,9,5,7,7,7,8,0,8,9,8,1,4,8,6,6,9,3,4,4,5,6,7,2,9,8,9,7,8,9,2,2,2,2,2,2,2,7,6,8,5,8,9,6,3,2,2,6,3,4,4,8,7,9,7,1,6,6,8,7,9,7,1,8,3,4,0,2,3,4,1,0,1,4,6,9,2,4,7,6,1,3,8,9,8,2,6,1,0,4,5,6,0,3,2,9,1,9,9,5,5,1,6,7,0,4,1,5,4,0,6,7,2,3,9,8,9,7,1,0,5,2,3,0,9,2,4,5,7,6,1,5,4,4,4,0,9,2,3,1,8,2,9,3,1,2,1,3,9,8,0,3,9,8,4,8,9,6,9,6,0,9,5,3,2,8,6,2,7,1,8,9,7,9,6,5,9,1,6,3,5,7,8,3,1,8,6,8,3,9,4,0,8,9,6,2,9,2,6,7,2,2,7,3,1,3,3,1,3,3,3,0,0,9,4,2,4,6,1,0,4,7,7,2,9,6,3,5,8,1,5,6,1,5,2,3,2,0,0,9,6,4,5,4,2,1,6,7,1,5,1,0,0,2,6,7,5,3,0,8,3,4,5,4,2,4,8,6,0,3,3,9,9,3,3,4,7,8,9,9,8,9,4,5,2,4,5,9,8,8,1,4,9,0,6,6,6,5,6,7,2,7,3,4,9,5,6,1,5,2,2,6,9,3,6,1,2,0,5,2,5,9,5,5,0,6,9,9,5,6,5,8,7,9,3,6,8,8,1,6,6,0,6,8,9,5,5,5,9,4,5,4,4,1,4,7,1,1,1,8,9,5,3,2,3,3,8,4,3,3,0,5,9,9,7,7,7,5,3,1,9,9,3,7,5,5,2,5,8,5,6,7,7,8,7,1,5,7,6,8,0,3,8,4,0,7,9,5,1,8,5,7,6,7,1,4,8,6,8,5,1,2,0,3,4,8,1,7,8,0,4,4,1,9,9,7,9,5,9,7,0,6,8,2,8,6,8,3,0,2,3,3,7,4,4,0,8,8,7,0,9,3,7,1,2,8,5,9,7,1,8,8,8,2,2,7,8,6,6,7,4,1,8,3,1,1,0,3,4,4,0,0,8,2,6,8,1,5,6,1,2,1,4,0,8,7,0,6,7,7,9,3,0,6,5,5,6,5,3,9,1,4,0,4,0,6,3,9,8,1,3,5,5,7,5,3,3,1,9,2,7,0,2,6,4,2,7,4,1,0,9,7,8,2,1,7,3,0,0,2,6,1,1,7,4,1,7,4,6,9,6,4,0,6,8,5,4,9,0,3,2,1,3,9,7,3,1,3,0,7,9,3,2,3,7,4,3,7,1,2,7,8,5,3,6,3,1,3,2,0,7,0,7,4,1,1,1,1,8,2,3,0,8,4,8,3,2,2,3,6,3,2,2,1,6,1,3,6,4,5,0,3,8,6,4,3,7,9,8,4,3,2,4,7,3,8,8,9,3,3,4,1,2,9,4,8,9,1,1,3,0,9,9,6,9,7,4,3,6,3,6,4,2,2,0,9,8,3,6,9,6,1,7,2,3,1,0,1,1,8,2,9,5,1,9,1,9,7,1,4,2,5,2,6,7,7,3,4,6,3,9,4,0,3,3,9,2,6,5,9,2,8,3,2,2,0,2,6,3,3,0,2,8,1,4,9,7,6,9,3,4,7,6,5,7,1,2,5,1,6,5,7,5,2,5,1,3,5,4,4,0,1,2,9,5,6,6,8,5,1,3,4,2,0,7,5,1,4,1,7,9,1,4,8,3,7,4,9,6,1,6,0,8,6,3,3,5,6,1,0,5,8,8,4,0,6,9,9,2,9,9,9,8,0,6,4,2,5,5,8,1,5,6,7,1,1,0,1,0,8,9,8,3,6,8,0,6,4,2,7,9,3,1,1,9,6,3,9,7,6,2,3,6,1,7,8,1,4,3,2,1,0,9,2,7,3,0,3,3,0,2,9,0,5,2,6,4,0,9,1,4,9,2,4,8,0,2,4,5,4,3,8,9,8,9,0,8,1,5,2,5,9,8,4,7,9,3,2,3,6,3,5,0,4,0,3,5,5,8,2,4,4,7,1,2,6,4,1,2,4,8,9,6,2,5,4,8,8,0,4,2,0,1,1,2,5,2,9,6,1,2,4,9,8,2,3,5,6,9,2,6,7,4,7,3,7,8,4,1,0,4,7,1,6,5,1,3,0,9,3,0,0,9,6,7,5,6,4,3,7,8,1,9,3,3,7,4,8,1,0,1,9,3,6,5,5,6,7,3,7,7,0,5,2,9,2,3,0,7,2,6,1,6,3,9,0,3,7,1,2,8,9,7,3,9,7,7,3,6,2,1,4,1,5,4,6,8,5,0,6,7,6,4,5,6,5,2,5,6,1,5,5,2,0,2,0,5,4,7,4,3,7,1,9,5,6,9,6,5,8,4,5,0,9,5,5,1,7,7,2,7,8,1,5,1,2,7,2,2,3,2,5,2,3,2,3,4,4,7,6,4,5,0,3,5,9,4,2,3,5,6,4,3,1,7,7,4,2,3,2,7,5,7,9,2,9,8,0,5,4,9,7,1,8,2,1,9,5,0,0,8,4,4,2,5,6,6,1,0,1,8,6,3,5,3,2,9,7,9,6,8,4,1,4,1,7,7,8,0,4,2,4,3,6,7,7,3,7,5,2,6,8,6,8,5,9,7,9,8,0,8,3,5,4,0,4,4,9,1,3,4,6,7,4,8,1,2,3,0,2,6,9,7,3,9,5,5,4,5,2,4,4,9,4,4,3,5,8,1,6,4,0,8,0,8,1,4,7,2,1,4,7,6,8,3,7,4,1,0,2,1,8,6,0,3,9,0,8,3,2,3,1,0,8,2,4,6,6,8,9,7,2,3,4,4,3,0,4,8,3,3,9,6,3,7,9,5,6,2,5,8,9,5,0,0,2,2,0,5,2,4,7,2,3,1,5,7,1,8,9,4,6,4,6,7,8,4,5,9,3,6,3,7,9,9,4,2,7,6,7,0,9,8,0,6,9,1,2,5,3,4,5,4,4,1,1,7,5,0,1,0,4,6,8,0,5,8,8,7,9,4,1,4,0,0,7,8,3,2,1,7,8,1,1,7,8,1,2,7,1,9,3,7,5,6,8,8,9,5,9,6,1,8,3,6,8,2,8,8,0,7,3,1,1,9,1,4,5,2,0,3,4,9,9,6,5,6,0,4,9,4,7,3,6,9,1,5,6,1,8,2,0,4,6,1,6,6,1,3,2,6,4,3,5,1,1,1,5,8,7,6,1,3,3,8,4,3,6,2,2,2,6,7,9,0,6,2,5,8,7,2,3,7,7,7,6,8,2,8,0,6,2,0,5,0,9,9,0,6,7,2,9,6,0,0,8,9,5,2,4,9,6,5,7,4,6,3,2,6,9,6,5,1,9,6,3,4,1,3,5,7,4,9,7,5,5,6,3,0,0,1,3,1,7,3,9,7,8,8,7,1,5,1,1,4,7,6,9,5,5,3,8,2,1,1,6,4,1,1,5,1,4,0,0,7,4,8,5,4,3,1,8,3,1,9,6,9,7,5,0,7,4,7,5,3,0,5,4,0,3,7,5,8,3,2,5,2,7,9,0,7,8,5,5,8,0,7,9,9,3,8,6,8,8,5,6,4,1,8,7,3,4,0,4,3,7,4,5,7,6,1,9,0,5,8,2,1,5,7,3,3,0,9,4,8,5,0,6,6,4,4,2,5,0,4,4,0,8,5,5,6,6,1,3,1,8,2,2,4,1,8,4,7,0,0,2,1,8,7,7,5,7,7,5,6,0,6,8,3,5,4,1,6,7,4,1,8,9,4,4,2,4,8,8,8,6,5,7,3,1,2,9,2,9,1,5,4,0,0,4,5,7,4,6,6,4,2,6,9,2,1,9,0,4,3,7,7,4,9,7,5,6,3,0,2,2,6,2,6,4,2,0,7,6,7,2,8,7,6,0,7,6,6,1,5,1,7,0,0,0,4,3,7,3,3,3,5,8,0,8,0,8,9,7,4,0,4,6,6,7,4,0,1,8,9,6,9,5,7,2,0,5,7,2,0,9,7,2,7,4,0,1,3,1,5,3,5,0,3,3,1,0,3,9,7,0,2,4,0,6,6,5,6,2,2,7,5,3,1,3,6,0,6,9,8,6,1,9,8,9,3,8,3,7,5,5,7,5,6,8,9,6,9,1,5,9,8,5,3,3,5,4,6,3,5,2,8,9,6,9,5,4,3,4,6,9,6,9,7,3,1,1,7,4,8,0,7,3,8,6,3,3,5,0,9,5,5,7,2,7,0,1,5,9,0,5,9,8,9,1,6,5,0,0,5,9,6,1,9,8,8,1,9,5,3,5,9,1,0,3,6,6,5,5,2,7,2,7,4,8,4,2,3,4,4,0,5,6,7,8,2,9,7,2,6,2,3,6,1,4,0,6,8,8,9,3,7,9,4,8,1,8,3,1,8,7,2,2,8,0,8,2,2,5,4,0,7,4,2,3,0,0,0,5,6,5,7,8,3,2,2,7,8,1,9,1,2,9,7,9,2,4,9,1,0,3,9,8,8,0,3,5,5,5,6,8,9,7,4,8,9,5,5,3,5,2,9,6,0,2,8,3,9,1,6,4,3,5,0,3,8,5,3,9,7,2,8,1,4,7,9,0,4,3,6,2,0,2,6,6,1,0,3,4,5,0,0,9,6,4,2,1,8,4,8,7,5,5,1,1,1,8,9,4,4,6,9,7,8,6,9,5,1,5,7,3,6,7,8,2,9,6,1,5,9,7,1,6,3,0,5,6,8,9,8,3,8,7,4,1,8,5,9,9,2,0,5,9,2,2,3,8,3,2,4,7,5,0,7,1,3,6,1,8,6,3,1,2,8,2,2,7,6,7,5,6,4,4,6,1,1,4,3,4,5,4,1,0,0,1,3,6,2,1,3,8,8,0,2,2,4,4,4,0,1,5,6,8,0,5,1,5,7,7,1,9,3,7,3,8,8,0,6,0,7,1,7,5,6,5,1,7,5,1,2,6,9,3,9,9,2,8,4,8,8,0,6,5,6,1,1,9,0,2,8,4,8,9,7,1,4,4,3,1,5,7,0,6,8,6,4,1,5,4,6,4,0,6,6,8,0,5,8,6,7,5,0,3,7,2,2,9,0,4,9,4,5,7,7,0,3,9,5,9,4,3,7,7,6,6,4,3,1,6,2,2,8,2,7,0,9,7,1,8,6,8,4,7,0,6,1,8,7,9,6,1,4,0,1,7,6,2,0,1,5,6,4,0,7,1,7,3,5,1,5,5,8,7,5,0,4,0,4,5,9,5,2,2,3,1,1,9,9,6,4,0,8,1,6,6,8,5,1,5,3,9,4,5,6,5,0,7,5,2,4,7,5,7,8,0,8,3,5,2,1,4,6,4,6,4,3,1,0,3,9,4,9,3,2,9,1,4,8,2,6,0,9,5,4,4,3,0,5,6,9,1,4,6,1,4,0,7,4,9,7,0,1,9,9,8,6,2,6,6,3,6,4,3,2,3,7,4,6,7,3,5,8,9,8,7,8,4,9,7,2,1,9,5,9,7,8,7,0,7,8,4,3,9,3,9,7,5,0,3,3,2,2,1,8,6,0,1,8,5,2,2,1,5,8,4,5,0,8,0,7,5,6,5,9,7,7,1,2,2,6,9,0,8,2,2,2,2,4,3,3,8,7,5,2,7,0,3,1,4,2,8,0,5,1,8,7,3,1,1,2,1,7,1,8,4,5,1,7,2,9,8,6,0,0,3,6,1,6,7,5,7,2,2,8,0,5,4,5,1,1,7,5,0,3,2,5,1,1,4,3,1,7,3,7,5,5,3,2,6,6,2,9,6,6,3,5,9,2,3,6,9,4,0,3,4,2,5,9,3,3,6,8,0,4,0,9,0,9,9,5,5,3,0,7,2,6,0,3,8,7,0,0,7,6,8,7,0,1,5,3,7,0,2,1,1,1,9,8,8,7,9,8,6,0,4,4,2,6,1,6,1,6,6,3,1,0,8,4,4,5,5,3,5,8,0,1,3,6,8,1,7,9,9,0,8,1,0,1,1,3,7,6,1,6,9,8,5,4,5,6,1,5,5,0,9,9,3,4,9,2,9,0,5,2,3,3,0,1,8,3,5,8,2,6,1,1,2,1,1,8,3,0,9,1,6,7,4,1,1,7,2,3,8,8,5,9,2,4,7,1,0,4,5,6,4,0,7,1,8,7,7,5,9,6,3,4,0,1,1,1,4,4,1,5,6,6,9,2,8,5,1,7,0,0,7,9,2,8,1,6,4,0,4,5,3,4,0,5,0,2,3,0,3,4,1,9,9,4,2,8,8,0,4,6,1,5,9,4,7,3,5,7,9,5,6,8,7,8,8,5,9,3,1,8,1,9,6,6,8,8,5,3,2,3,0,0,1,2,2,8,1,4,7,2,4,7,0,1,4,3,8,9,3,0,5,6,5,5,8,1,5,8,6,8,2,4,5,2,9,5,6,9,7,8,8,7,1,2,0,6,2,9,7,0,5,1,5,2,1,8,1,2,5,0,4,4,0,4,6,9,9,0,7,2,9,1,9,6,0,1,2,4,0,6,7,9,8,1,7,1,4,9,7,5,3,4,8,5,2,4,6,1,6,9,7,0,6,1,4,4,4,4,8,4,7,0,6,2,6,1,0,3,7,5,4,4,7,8,8,9,9,8,8,2,7,6,7,9,9,2,1,1,3,1,9,7,3,3,6,9,8,1,1,2,4,9,2,3,6,3,8,6,4,2,7,3,9,4,0,9,8,8,1,9,9,8,8,9,0,8,9,7,7,3,9,3,8,5,7,2,3,9,7,9,7,7,8,7,0,3,9,7,4,8,7,7,1,1,0,3,5,0,6,0,6,1,8,6,6,4,0,1,3,2,0,9,5,9,1,1,6,1,0,4,4,7,1,6,4,0,2,7,3,0,2,3,8,1,1,6,5,5,9,8,5,6,8,1,0,8,6,7,8,6,7,9,7,7,3,7,7,9,0,3,5,4,1,3,4,9,8,8,8,1,8,2,0,5,1,1,6,9,6,4,8,8,7,6,7,9,8,6,6,4,2,3,9,2,4,2,5,3,2,6,6,4,2,7,6,0,4,7,3,5,9,3,8,1,8,9,5,0,5,1,4,4,6,5,1,6,0,8,6,3,7,4,9,0,9,3,6,7,7,5,9,7,1,0,8,5,0,0,8,3,3,4,6,2,2,1,4,0,6,4,5,6,9,6,1,2,6,8,5,6,8,2,2,7,6,2,7,9,3,6,4,0,5,3,4,5,0,6,6,0,3,2,6,1,5,4,7,9,3,4,0,3,5,3,9,4,9,8,9,3,7,0,3,3,8,3,1,8,9,1,0,0,9,3,2,2,9,1,0,3,8,5,2,5,8,9,5,8,4,1,8,3,0,6,3,9,3,7,7,3,0,8,0,9,1,9,5,3,2,5,7,1,4,5,2,3,5,4,4,5,1,9,8,2,7,9,4,2,1,2,4,2,4,1,0,7,0,1,0,4,4,6,3,2,4,3,7,1,4,9,2,3,3,8,2,5,1,8,4,2,2,2,8,3,3,1,9,8,4,5,5,2,6,3,3,2,3,0,0,3,1,3,5,3,3,5,6,9,2,3,5,3,6,3,6,5,1,7,4,9,3,9,6,5,2,8,3,7,9,2,6,6,2,0,2,1,0,3,1,6,3,9,5,3,9,6,7,6,3,9,1,6,9,0,1,1,5,6,3,0,7,2,6,3,0,2,4,1,1,4,4,9,6,3,5,8,1,1,8,6,5,0,8,1,9,3,7,1,3,5,2,7,5,4,8,3,2,6,9,7,1,3,5,3,3,8,9,9,1,2,6,5,9,5,0,7,5,4,8,0,6,5,5,9,7,6,5,0,0,6,2,1,4,8,8,4,7,0,8,7,7,7,1,6,6,9,8,3,6,5,2,5,0,4,1,9,0,4,2,7,7,5,3,9,8,8,2,9,0,7,2,9,4,7,8,5,2,2,0,0,9,1,4,3,4,9,0,5,3,1,1,7,3,4,4,4,5,5,0,2,4,9,5,4,1,4,9,7,8,5,0,9,9,5,3,8,2,2,9,1,3,4,3,0,3,1,8,6,7,2,3,9,2,5,2,0,4,3,7,6,2,2,4,4,1,0,6,9,5,2,2,6,3,3,3,7,8,7,5,8,2,8,6,1,2,5,2,7,1,3,2,6,5,0,7,4,7,0,3,7,4,1,4,2,7,7,1,6,3,8,0,0,1,5,3,8,6,9,3,1,5,0,7,8,3,9,8,9,4,8,0,6,1,1,6,8,7,0,7,6,4,4,4,1,8,9,9,9,8,0,0,9,7,5,8,1,1,8,1,3,0,0,9,7,8,6,1,0,6,0,3,5,3,6,5,3,3,2,3,3,6,8,5,5,0,5,2,1,2,7,0,7,5,3,3,6,4,1,3,5,8,1,0,6,7,3,9,4,6,4,2,8,4,3,1,7,3,6,5,6,0,6,9,3,5,1,3,5,7,7,5,1,7,0,8,3,0,1,9,7,1,8,7,1,4,5,7,5,6,6,1,3,6,6,8,4,5,8,9,0,8,0,7,7,4,9,6,3,1,6,1,9,9,6,8,1,0,0,4,1,3,4,8,8,9,6,9,0,1,4,4,6,9,4,8,4,3,7,7,4,5,8,1,8,7,7,3,4,3,0,3,2,5,1,7,8,1,0,8,5,6,3,7,1,6,0,9,8,6,7,8,5,7,5,6,6,9,2,0,4,6,8,5,9,4,6,3,8,8,3,1,2,2,0,4,4,0,8,8,1,9,9,1,7,9,1,7,0,5,9,6,5,1,0,4,2,6,0,3,3,9,1,4,9,0,7,0,8,0,4,1,9,5,1,0,3,6,0,3,1,2,8,4,6,7,6,1,2,6,5,5,5,3,9,8,0,0,7,3,5,4,1,1,5,3,8,4,0,0,4,6,8,3,9,8,5,7,7,7,3,9,9,2,4,6,2,0,9,6,3,0,7,4,2,3,0,2,5,5,8,1,7,5,8,4,7,3,8,2,5,3,8,8,6,3,5,7,7,8,1,3,7,6,9,5,1,8,4,0,6,6,4,9,3,2,2,9,2,6,2,9,9,7,2,9,6,7,8,8,1,6,0,9,0,4,8,9,7,6,9,2,7,5,3,9,8,7,9,9,8,4,4,6,9,5,5,0,6,4,9,2,3,7,6,2,7,9,1,2,1,8,9,6,5,4,5,7,5,1,7,0,9,1,9,9,9,9,2,1,9,6,0,7,1,7,6,7,7,5,3,8,9,2,3,6,4,2,7,0,8,0,6,7,5,2,9,6,5,3,1,5,2,0,9,3,6,8,0,2,6,8,4,6,1,1,1,5,0,5,3,2,9,1,3,8,0,5,9,2,0,0,5,8,4,5,2,8,7,2,4,3,0,4,5,2,8,5,8,3,6,1,2,9,8,3,5,5,3,3,1,5,0,7,5,2,1,1,4,9,5,2,3,8,8,1,6,6,9,1,0,5,2,7,8,0,8,5,7,6,5,9,8,5,1,1,1,1,5,2,2,6,9,3,2,6,6,1,7,4,2,2,9,9,0,7,1,5,1,8,7,5,0,7,5,2,1,3,3,4,7,3,8,8,6,1,6,6,8,4,4,3,2,1,6,1,0,2,1,2,4,5,8,8,0,5,9,2,7,3,5,0,9,5,5,7,8,8,8,2,5,8,6,6,9,4,5,5,0,1,9,3,9,5,1,5,3,2,2,6,7,3,3,6,2,9,0,8,6,2,6,9,1,2,0,2
    auto check = [](std::vector<int> const& in, std::size_t k = 1){
        std::vector<int> result{in};
        std::vector<int> expected{in};
        std::vector<int> c{in};
        auto start_time{std::chrono::high_resolution_clock::now()};
        std::ranges::rotate(expected, std::end(expected) - (k % std::size(expected)) );
        auto end_time{std::chrono::high_resolution_clock::now()};
        std::cout << "ranges rotate time " << (end_time - start_time).count() << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::rotate(std::begin(c), std::end(c) - (k % std::size(in)), std::end(c));
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "stdlib rotate time " << (end_time - start_time).count() << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        Solution{}.rotate(result, k);
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "my rotate time " << (end_time - start_time).count() << std::endl;
        EXPECT_EQ(result, expected);
    };

    check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 11);
    check({0});
    check({0, 1});
    check({0, 1, 2, 3, 4}, 3);
    check({0, 1, 2});
    check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    for (int i{}; i < 99; i++) {
        check({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, i);
    }
    check({0, 1, 2}, 2);

    check({MEGA_VALUE}, 99'999);
}

TEST(log2_lowerbound, main) {
    auto check = [](auto const& input, auto const& target) {
        return Solution::log_search_lowerbound(std::begin(input), std::end(input), target);
    };

    std::vector<int> input;
    input = {1, 2, 3, 4, 4, 4, 5, 6};
    EXPECT_EQ( check(input, 4), std::next(std::begin(input), 3));
    EXPECT_EQ( check(input, 10), std::end(input));
    EXPECT_EQ( check(input, 0), std::end(input));
    input = {1, 2, 3, 4, 5};
    EXPECT_EQ( check(input, 1), std::next(std::begin(input), 0));
    EXPECT_EQ( check(input, 6), std::next(std::begin(input), 5));
    EXPECT_EQ( check(input, 5), std::next(std::begin(input), 4));
}

TEST(log2_upperbound, main) {
    auto check = [](auto const& input, auto const& target) {
        return Solution::log_search_upperbound(std::begin(input), std::end(input), target);
    };

    std::vector<int> input;
    input = {1, 2, 3, 4, 4, 4, 5, 6};
    EXPECT_EQ( check(input, 4), std::next(std::begin(input), 5));
    EXPECT_EQ( check(input, 10), std::end(input));
    EXPECT_EQ( check(input, 0), std::end(input));
    input = {1, 2, 3, 4, 5};
    EXPECT_EQ( check(input, 1), std::next(std::begin(input), 0));
    EXPECT_EQ( check(input, 6), std::next(std::begin(input), 5));
    EXPECT_EQ( check(input, 5), std::next(std::begin(input), 4));
}

TEST(move_zeroes, main) {
    auto check = [](std::vector<int> in) {
        auto result{in};
        auto expected{in};
        check_time(&Solution::moveZeroes, result);
        check_time(std::ranges::sort, expected, [](auto const& a, auto const& b){ return !a ? false : !b ? true : a < b;});

        EXPECT_EQ( expected, result);
    };

    check({});
    check({0});
    check({1, 2, 3, 4, 5});
    check({1, 0, 2, 0, 3});
    check({1, 0, 2, 0, 3, 0, 4, 0, 5});
    check({1, 0, 2, 0, 3, 0, 4, 0, 5, 0,});
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
