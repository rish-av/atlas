import random

def generate_file_content(file_name):
    if file_name in file_to_functions:
        funcs = file_to_functions[file_name]
        func_contents = []
        for f in funcs:
            if f in function_to_lines:
                func_contents.append("\n".join(function_to_lines[f]))
            else:
                func_contents.append(f"def {f}():\n    pass")
        # Add a simple header to simulate full file content
        header = f"# File: {file_name}\nimport os\n\n"
        return header + "\n\n".join(func_contents)
    else:
        return "def default_func():\n    pass"

def generate_function_definitions(file_name):
    if file_name in file_to_functions:
        funcs = file_to_functions[file_name]
        func_defs = []
        for f in funcs:
            if f in function_to_lines:
                func_defs.append("\n".join(function_to_lines[f]))
            else:
                func_defs.append(f"def {f}():\n    pass")
        return func_defs
    else:
        return ["def default_func():\n    pass"]

# Realistic bug reports
realistic_bug_reports = [
    "Application crashes on login when using invalid credentials",
    "Database connection fails with timeout error",
    "Profile page does not load due to missing user data",
    "Logout button does not trigger logout",
    "Error calculating total price in checkout",
    "Unable to update user details in profile",
    "Payment processing error in billing module",
    "File upload fails with unexpected error",
    "Search functionality returns no results",
    "Email notification not sent on registration"
]

# Candidate file names
candidate_files_pool = [
    "auth.py", "database.py", "views.py", "checkout.py", "billing.py",
    "utils.py", "profile.py", "data_service.py", "notification.py", "upload.py",
    "network.py", "server.py", "search.py", "indexer.py"
]

# Mapping from file to functions
file_to_functions = {
    "auth.py": ["login", "logout", "validate_user", "encrypt_password"],
    "database.py": ["connect_db", "query_db", "update_record", "close_connection"],
    "views.py": ["render_home", "render_profile", "render_error", "handle_request"],
    "checkout.py": ["calculate_total", "apply_discount", "process_payment", "validate_cart"],
    "billing.py": ["generate_invoice", "process_payment", "update_billing", "send_receipt"],
    "utils.py": ["format_date", "log_error", "parse_json", "calculate_discount"],
    "profile.py": ["load_profile", "update_profile", "validate_profile", "delete_profile"],
    "data_service.py": ["fetch_data", "update_data", "delete_data", "insert_data"],
    "notification.py": ["send_email", "send_sms", "schedule_notification", "log_notification"],
    "upload.py": ["upload_file", "validate_file", "store_file", "generate_file_id"],
    "network.py": ["connect", "disconnect", "send_data", "receive_data"],
    "server.py": ["start_server", "handle_request", "shutdown_server", "restart_server"],
    "search.py": ["perform_search", "filter_results", "sort_results", "paginate"],
    "indexer.py": ["build_index", "update_index", "delete_index", "search_index"]
}

# Mapping from function to full function definitions (lines)
function_to_lines = {
    "login": [
        "def login(request):",
        "    username = request.get('username')",
        "    password = request.get('password')",
        "    user = authenticate(username, password)",
        "    if user:",
        "        return redirect('home')",
        "    else:",
        "        return error()"
    ],
    "logout": [
        "def logout(request):",
        "    session.clear()",
        "    return redirect('login')"
    ],
    "validate_user": [
        "def validate_user(user):",
        "    if not user.is_active:",
        "        return False",
        "    return True"
    ],
    "encrypt_password": [
        "def encrypt_password(password):",
        "    return hash_function(password)"
    ],
    "connect_db": [
        "def connect_db():",
        "    connection = db.connect()",
        "    return connection"
    ],
    "query_db": [
        "def query_db(query):",
        "    results = connection.execute(query)",
        "    return results"
    ],
    "update_record": [
        "def update_record(record):",
        "    db.update(record)",
        "    return True"
    ],
    "close_connection": [
        "def close_connection():",
        "    connection.close()"
    ],
    "render_home": [
        "def render_home(request):",
        "    return render(request, 'home.html')"
    ],
    "render_profile": [
        "def render_profile(request):",
        "    return render(request, 'profile.html')"
    ],
    "render_error": [
        "def render_error(request):",
        "    return render(request, 'error.html')"
    ],
    "handle_request": [
        "def handle_request(request):",
        "    process(request)",
        "    return response"
    ],
    "calculate_total": [
        "def calculate_total(items):",
        "    total = sum(item.price for item in items)",
        "    return total"
    ],
    "apply_discount": [
        "def apply_discount(total, discount):",
        "    return total - (total * discount)"
    ],
    "process_payment": [
        "def process_payment(payment_info):",
        "    if validate(payment_info):",
        "        charge(payment_info)",
        "        return True",
        "    return False"
    ],
    "validate_cart": [
        "def validate_cart(cart):",
        "    return len(cart) > 0"
    ],
    "generate_invoice": [
        "def generate_invoice(order):",
        "    invoice = create_invoice(order)",
        "    return invoice"
    ],
    "update_billing": [
        "def update_billing(details):",
        "    billing.update(details)",
        "    return True"
    ],
    "send_receipt": [
        "def send_receipt(email, invoice):",
        "    email.send(invoice)"
    ],
    "format_date": [
        "def format_date(date):",
        "    return date.strftime('%Y-%m-%d')"
    ],
    "log_error": [
        "def log_error(error):",
        "    logger.error(error)"
    ],
    "parse_json": [
        "def parse_json(data):",
        "    return json.loads(data)"
    ],
    "calculate_discount": [
        "def calculate_discount(price, discount):",
        "    return price * discount"
    ],
    "load_profile": [
        "def load_profile(user_id):",
        "    profile = database.get_profile(user_id)",
        "    return profile"
    ],
    "update_profile": [
        "def update_profile(user_id, data):",
        "    database.update_profile(user_id, data)",
        "    return True"
    ],
    "validate_profile": [
        "def validate_profile(profile):",
        "    return profile is not None"
    ],
    "delete_profile": [
        "def delete_profile(user_id):",
        "    database.delete_profile(user_id)"
    ],
    "fetch_data": [
        "def fetch_data(query):",
        "    return database.fetch(query)"
    ],
    "update_data": [
        "def update_data(query, data):",
        "    database.update(query, data)"
    ],
    "delete_data": [
        "def delete_data(query):",
        "    database.delete(query)"
    ],
    "insert_data": [
        "def insert_data(data):",
        "    database.insert(data)"
    ],
    "send_email": [
        "def send_email(to, subject, body):",
        "    email.send(to, subject, body)"
    ],
    "send_sms": [
        "def send_sms(number, message):",
        "    sms.send(number, message)"
    ],
    "schedule_notification": [
        "def schedule_notification(notification):",
        "    scheduler.schedule(notification)"
    ],
    "log_notification": [
        "def log_notification(notification):",
        "    logger.info(notification)"
    ],
    "upload_file": [
        "def upload_file(file):",
        "    storage.save(file)",
        "    return True"
    ],
    "validate_file": [
        "def validate_file(file):",
        "    return file.size < MAX_SIZE"
    ],
    "store_file": [
        "def store_file(file):",
        "    storage.store(file)"
    ],
    "generate_file_id": [
        "def generate_file_id(file):",
        "    return hash(file)"
    ],
    "connect": [
        "def connect(host):",
        "    connection = socket.connect(host)",
        "    return connection"
    ],
    "disconnect": [
        "def disconnect(connection):",
        "    connection.close()"
    ],
    "send_data": [
        "def send_data(connection, data):",
        "    connection.send(data)"
    ],
    "receive_data": [
        "def receive_data(connection):",
        "    return connection.receive()"
    ],
    "start_server": [
        "def start_server():",
        "    server = Server()",
        "    server.listen()"
    ],
    "shutdown_server": [
        "def shutdown_server():",
        "    server.shutdown()"
    ],
    "restart_server": [
        "def restart_server():",
        "    shutdown_server()",
        "    start_server()"
    ],
    "perform_search": [
        "def perform_search(query):",
        "    results = search_engine.search(query)",
        "    return results"
    ],
    "filter_results": [
        "def filter_results(results):",
        "    return [r for r in results if r.is_valid()]"
    ],
    "sort_results": [
        "def sort_results(results):",
        "    return sorted(results, key=lambda x: x.rank)"
    ],
    "paginate": [
        "def paginate(results, page, per_page):",
        "    start = (page - 1) * per_page",
        "    return results[start:start+per_page]"
    ],
    "build_index": [
        "def build_index(documents):",
        "    index = create_index(documents)",
        "    return index"
    ],
    "update_index": [
        "def update_index(index, document):",
        "    index.update(document)"
    ],
    "delete_index": [
        "def delete_index(index, doc_id):",
        "    index.remove(doc_id)"
    ],
    "search_index": [
        "def search_index(index, query):",
        "    return index.search(query)"
    ]
}

class sample_data():
    def __init__(self):
        trial_dataset = []
        for i in range(50):
            bug_report = random.choice(realistic_bug_reports)
            candidate_file_names = random.sample(candidate_files_pool, 3)
            candidate_files = [generate_file_content(fname) for fname in candidate_file_names]
            correct_file_idx = random.randint(0, 2)
            correct_file_name = candidate_file_names[correct_file_idx]
            
            candidate_functions = generate_function_definitions(correct_file_name)
            correct_function_idx = random.randint(0, len(candidate_functions) - 1)
            candidate_lines = candidate_functions[correct_function_idx].splitlines()
            
            num_lines = len(candidate_lines)
            correct_line_labels = [0] * num_lines
            num_buggy = random.choice([1, 2]) if num_lines > 0 else 1
            buggy_indices = random.sample(range(num_lines), num_buggy)
            for idx in buggy_indices:
                correct_line_labels[idx] = 1

            sample = {
                'bug_report': f"Sample {i+1}: {bug_report}",
                'candidate_files': candidate_files,
                'correct_file_idx': correct_file_idx,
                'candidate_functions': candidate_functions,
                'correct_function_idx': correct_function_idx,
                'candidate_lines': candidate_lines,
                'correct_line_labels': correct_line_labels
            }
            trial_dataset.append(sample)

        self.train_dataset = trial_dataset[:80]
        self.test_dataset = trial_dataset[80:]
