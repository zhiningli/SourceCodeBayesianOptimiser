from typing import List, Optional
" Question 1"


def countUnpaired(arr: List) -> int:
    unmatched_set = set()

    for ele in arr:
        if ele not in unmatched_set:
            unmatched_set.add(ele)
        else:
            unmatched_set.remove(ele)

    return len(unmatched_set)

print(countUnpaired([1,2,3,4]))
""" 
How would you best solve this in the case where the input would be provided as a stream of values, versus the entire input being available to you in memory?

I will change this method into a class so that I can instantiate a persistent instance and retain the ability to track unpaired numbers in the unmatched_set
"""

class UnpairedTracker:

    def __init__(self, arr: Optional[List] = None):
        self.unmatched_set = set()
        if arr:
            self.process_arr(arr)
    
    def process_arr(self, arr: List) -> None:
        for ele in arr:
            if ele not in self.unmatched_set:
                self.unmatched_set.add(ele)
            else:
                self.unmatched_set.remove(ele)
    
    def countUnpaired(self, arr: Optional[List] = None) -> int:
        if arr:
            self.process_arr(arr)
        return len(self.unmatched_set)


""" Question 2
In the question setting, I will implicitly assume that all rooms are connected to room 0 via some routes
"""

from collections import deque, defaultdict

def furthestRoom(doors: Optional[List]) -> int:

    graph = defaultdict(list)
    for roomA, roomB in doors:
        graph[roomA].append(roomB)
        graph[roomB].append(roomA)

    visited = set([0])
    queue = deque([(0, 0)])
    max_distance = 0
    while queue:
        curr_room, dist = queue.popleft()
        max_distance = max(max_distance, dist)

        for neighbor in graph[curr_room]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist+1))
    return max_distance

print(furthestRoom([(0,1),(1,2),(1,3),(2,4),(4,5),(3,4)]))
        
""" Question 3

Some basic math for how many UUIDs can be stored on a micro-controller, assume there are 5MB memory avaialble for storage

one UUID takes 16 bytes, 1 megabype can store 65 536 UUIDs,

As a pessimistic estimation factoring in overhead (pointers) and addtional helper methods, 5MB memory can store 200 000 UUID.
I believe it is highly unlikely in real-world scenarios that so many UUIDs are stored through a single electronic door's micro-controller.

Hence, I would use a simple ordered set from C++ standard library. At low level, an ordered map is implemented as a balanced binary
search tree. It sorts the UUIDs and hence provide O(logN) time complexity for look-up. The insertion and deletion upon commumnication with
cloud is O(logN) as well, making it extremely fast compared to other in memory data-structure.

Another canditate is an unordered set (hashtable) also from C++ standard library, it offers constant time complexity for insertion, look up/deletion on average but takes worst case
O(N) time complexity for all insertion, look up and deletion. Given the importance of frequent and efficient access checks, an ordered set is preferable, as it guarantees predictable 
performance and avoids the rare but significant risks associated with hash value collisions.

If implemented as an ordered set and assume a 32 bit system (4 bytes per pointer), each UUID nodes will take 16 bytes + 3 * 4 bytes = 28 bytes. 
5MB can store 180 000 + UUIDs which is great :).

In the rare case of extremely large datasets that exceeding memory limit, I would adopt sharding based on location, department or other logical criterion.
The last resort is to use a custom hash table designed to minimise collisions while balance memory and performance.

I will not consider using a bloom filter for faster loop up due to its inherent false positives, which pose a significant security risk by potentially allowing unautorised access.
I want to be risk averse when designing an access control system and the trade-off between faster lookips and the possibility of false positives is undesirable. 

Therefore, a set from C++ STL is my preferred solution for its simplicity, extremly fast look up time, fast insertion and deletion. The set datastructure can be 
easily fitted into the memory of a low powered micro-controller while granting access to 180 000+ UUIDs.

"""
