import pandas as pd

# Load the CSV file
file_path = input()
data = pd.read_csv(file_path)

# Initialize variables to store results
attacker_ip = None
sql_injection_attempts = 0
first_payload = None
last_payload = None
payloads_with_colon = 0

# Define SQL injection keywords
sql_keywords = ["union", "select", "' or", "' and", "--", "'=", "'&"]

# Iterate through the DataFrame to analyze SQL injection attempts
for index, row in data.iterrows():
    # Check if the packet contains SQL injection keywords in the 'Info' column
    info_lower = row['Info'].lower()
    if any(keyword in info_lower for keyword in sql_keywords):
        # Count SQL injection attempts
        sql_injection_attempts += 1
        
        # Extract attacker IP (assuming a single attacker)
        if attacker_ip is None:
            attacker_ip = row['Source']
        
        # Extract URI from packet details (Info column)
        uri_start = row['Info'].find('/')
        uri_end = row['Info'].find('HTTP') if 'HTTP' in row['Info'] else len(row['Info'])
        uri = row['Info'][uri_start:uri_end].strip() if uri_start != -1 else None
        
        # Check for payloads with formatting symbols (colon ':') specifically in the URI
        if ':' in info_lower or '0x3a' in info_lower:
            payloads_with_colon += 1
        
        # Update first and last payloads based on timestamp
        if first_payload is None or float(row['Time']) < float(first_payload[1]):
            first_payload = (uri, row['Time'])
        if last_payload is None or float(row['Time']) > float(last_payload[1]):
            last_payload = (uri, row['Time'])

# Prepare output values
attacker_ip_output = f"1A: {attacker_ip}" if attacker_ip else "1A: NULL"
sql_attempts_output = f"2A: {sql_injection_attempts}"
first_payload_output = f"3A: {first_payload[0]}" if first_payload else "3A: NULL"
last_payload_output = f"4A: {last_payload[0]}" if last_payload else "4A: NULL"
colon_payload_count_output = f"5A: {payloads_with_colon}"

# Print results
print(attacker_ip_output)
print(sql_attempts_output)
print(first_payload_output)
print(last_payload_output)
print(colon_payload_count_output)


'''
Expected Output for one Visible Test Case
1A: 10.0.2.5
2A: 20
3A: /dvwa/vulnerabilities/sqli/?id=2'&Submit=Submit
4A: /dvwa/vulnerabilities/sqli/?id=2'+union+select+group_concat(user_id,0x3a,user,0x3a,password),2+from+users--+&Submit=Submit
5A: 2
'''