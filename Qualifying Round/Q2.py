import hashlib

def compute_sha256_hash(data):
    """
    Computes the SHA-256 hash of the given data (string).
    Returns the hash as a hexadecimal string.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def compute_merkle_root(transactions):
    """
    Computes the Merkle root for the given list of transactions.
    """
    if not transactions:
        return ""

    if len(transactions) == 1:
        # If only one transaction, return its SHA-256 hash directly
        return compute_sha256_hash(transactions[0])

    # Step 1: Handle input transactions (already hashed or raw strings)
    current_level = [
        tx if tx.startswith("0x") else compute_sha256_hash(tx)
        for tx in transactions
    ]

    # Step 2: Perform recursive pairing and hashing until one hash remains
    while len(current_level) > 1:
        if len(current_level) % 2 != 0:
            # Duplicate the last hash if odd number of hashes
            current_level.append(current_level[-1])

        # Pair and hash
        next_level = []
        for i in range(0, len(current_level), 2):
            combined = current_level[i] + current_level[i + 1]
            next_level.append(compute_sha256_hash(combined))

        current_level = next_level

    return current_level[0]  # The last remaining hash is the Merkle root

def notmain():
    """
    Reads input, computes Merkle roots for test cases, and prints the results.
    """
    T = int(input().strip())  # Number of test cases
    results = []

    for _ in range(T):
        N = int(input().strip())  # Number of transactions
        transactions = [input().strip() for _ in range(N)]

        # Compute Merkle root for this test case
        merkle_root = compute_merkle_root(transactions)
        results.append(merkle_root)

    # Print all results
    for result in results:
        print(result)

notmain()


'''

# Visible Test Case
Input: 0x3a6d79e019b10e459d2624f38f24c5ead4db463bc544fec1c53aef85c4f85774
Expected Output: aa17c79e02cbd20473e0e26e1cd110242a29d8e4a43d1c6b07a6b2037566aa54


# Hidden Test Case
Input: 
0x3a6d79e019b10e459d2624f38f24c5ead4db463bc544fec1c53aef85c4f85774
0x169c7efe0b04a3a0bd7a34f966f40a4ef7e087c76b88d4c93c84622253971be7
0x0815794b25543b2d7d5e575d3b441e7c79370a2577023894fd1bf81251371a5a
Expected Output: 1ca2374bfe088e2653104271975d0e5b4c4571ea82f44eb91dcb996d4c9d3eab

'''