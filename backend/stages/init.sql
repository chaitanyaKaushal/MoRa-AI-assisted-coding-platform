-- Create problems table
CREATE TABLE IF NOT EXISTS problems (
    id SERIAL PRIMARY KEY,
    problem_id VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    problem_description TEXT NOT NULL,
    examples TEXT NOT NULL,
    constraints TEXT NOT NULL,
    constraint_source VARCHAR(50),
    tags TEXT[],
    starter_code TEXT NOT NULL,
    evaluation_type VARCHAR(50) NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create test_cases table
CREATE TABLE IF NOT EXISTS test_cases (
    id SERIAL PRIMARY KEY,
    problem_id VARCHAR(50) NOT NULL,
    test_inputs JSONB NOT NULL,
    expected_output JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(problem_id) ON DELETE CASCADE
);

-- Create reference_solutions table
CREATE TABLE IF NOT EXISTS reference_solutions (
    id SERIAL PRIMARY KEY,
    problem_id VARCHAR(50) UNIQUE NOT NULL,
    solution_code TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(problem_id) ON DELETE CASCADE
);

-- Create submissions table
CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(50) UNIQUE NOT NULL,
    problem_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    user_code TEXT NOT NULL,
    passed_tests INTEGER,
    total_tests INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(problem_id) ON DELETE CASCADE
);

-- Create submission_results table
CREATE TABLE IF NOT EXISTS submission_results (
    id SERIAL PRIMARY KEY,
    submission_id VARCHAR(50) NOT NULL,
    test_number INTEGER,
    input JSONB,
    expected JSONB,
    actual JSONB,
    passed BOOLEAN,
    error_message TEXT,
    FOREIGN KEY (submission_id) REFERENCES submissions(submission_id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_problems_problem_id ON problems(problem_id);
CREATE INDEX IF NOT EXISTS idx_test_cases_problem_id ON test_cases(problem_id);
CREATE INDEX IF NOT EXISTS idx_submissions_problem_id ON submissions(problem_id);
CREATE INDEX IF NOT EXISTS idx_submissions_user_id ON submissions(user_id);