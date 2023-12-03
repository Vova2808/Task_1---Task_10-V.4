##################################################################################
# Task_1
# 1. Напишите программу, которая принимает
# текст и выводит два слова: наиболее часто
# встречающееся и самое длинное.

def Task_1():
    import collections

    text = 'lorem ipsum dolor sit amet amet amet'
    words = text.split()
    counter = collections.Counter(words)
    most_common, occurrences = counter.most_common()[0]

    longest = max(words, key=len)
    print(most_common, longest)

# Task_11()

# Well, the average
# :) or :| or :(



# Two Sum

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n_dict = {}
        for i in range(len(nums)):
            f_num = target - nums[i]
            if n_dict.get(f_num) != None:
                return [n_dict[f_num], i]
            n_dict[nums[i]] = i
        return []

# 7. Reverse Integer

class Solution:
    def reverse(self, x: int) -> int:
        res = int(str(abs(x))[::-1])
        if res > 2 ** 31 - 1:
            return 0
        if x < 0:
            return -res
        return res


# 9. Palindrome Number

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if (x < 0) | ((x % 10 == 0) & (x != 0)):
            return False
        rx = 0
        while rx < x:
            rx = rx * 10 + x % 10
            x //= 10
        return (x == rx) | (x == rx // 10)

# 13. Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        sym_val = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000 }
        res = 0
        prev = 0
        for sym in s:
            cur = sym_val[sym]
            mult = 1
            if cur > prev:
                mult = -1
            res += prev * mult
            prev = cur
        return res + prev

# 14. Longest Common Prefix

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ''
        if len(strs) == 0:
            return res
        for ltr in strs[0]:
            for wrd in strs[1::]:
                if wrd.find(res + ltr) != 0:
                    return res
            res += ltr
        return res

# 20. Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in s:
            if char in brackets:
                stack.append(char)
            else:
                if not stack or brackets[stack[-1]] != char:
                    return False
                stack.pop()
        return not stack

# 26. Remove Duplicates from Sorted Array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        i = 0
        for j in range(1, len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        return i + 1

# 27. Remove Element

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for j in range(len(nums)):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i

# 28. Implement strStr()

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle or haystack == needle:
            return 0
        lh = len(haystack)
        ln = len(needle)
        for i in range(lh - ln + 1):
            if haystack[i] == needle[0] and haystack[i:i+ln] == needle:
                return i
        return -1

# 35. Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                high = mid - 1
            else:
                low = mid + 1
        return low

# 38. Count and Say

class Solution:
    def countAndSay(self, n: int) -> str:
        if n <= 1:
            return '1'
        else:
            term = self.countAndSay(n - 1)
            res, cnt, say = '', 1, term[0]
            for i in range(1, len(term)):
                if term[i] != say:
                    res += f'{cnt}{say}'
                    cnt, say = 1, term[i]
                else:
                    cnt += 1
            return f'{res}{cnt}{say}'

# 54. Spiral Matrix

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        cur = {'i': 0, 'j': 0}
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dir_steps = [m, n]
        res, dir_num, step = [], 0, 0
        for _ in range(m * n):
            step += 1
            res.append(matrix[cur['i']][cur['j']])
            if step == dir_steps[(dir_num + 1) % 2]:
                dir_steps[dir_num % 2] -= 1
                step = 0
                dir_num += 1
            cur['i'] += dirs[dir_num % 4][0]
            cur['j'] += dirs[dir_num % 4][1]
        return res

# 58. Length of Last Word

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        cnt = 0
        for i in reversed(range(len(s.rstrip()))):
            if s[i] != ' ':
                cnt += 1
            else:
                break
        return cnt

# 59. Spiral Matrix II

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        mat = [[0] * n for _ in range(n)]
        cur = {'i': 0, 'j': 0}
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dir_steps, dir_num, step = [n, n], 0, 0
        for i in range(1, n * n + 1):
            mat[cur['i']][cur['j']] = i
            if i - step == dir_steps[(dir_num + 1) % 2]:
                dir_steps[dir_num % 2] -= 1
                step = i
                dir_num += 1
            cur['i'] += dirs[dir_num % 4][0]
            cur['j'] += dirs[dir_num % 4][1]
        return mat

# 67. Add Binary

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(a) > len(b):
            b = b.zfill(len(a))
        elif len(a) < len(b):
            a = a.zfill(len(b))
        mem, res = 0, ''
        for i in reversed(range(len(a))):
            dsum = int(a[i]) + int(b[i]) + mem
            res += str(dsum % 2)
            mem = dsum // 2
        if mem:
            res += str(mem)
        return res[::-1]

