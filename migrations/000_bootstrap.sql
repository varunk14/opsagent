-- Runs once, on first container start, via docker-entrypoint-initdb.d.
--
-- A separate database for tests so a test run can never truncate dev data.
-- Tests connect to opsagent_test; everything else uses opsagent.
CREATE DATABASE opsagent_test;
