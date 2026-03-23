use rayon::ThreadPoolBuilder;

pub struct TaskSystem {
    pool: rayon::ThreadPool,
}

impl TaskSystem {
    pub fn new() -> Self {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()
            .expect("Failed to create thread pool");

        Self { pool }
    }
}

impl Default for TaskSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskSystem {

    pub fn spawn<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.spawn(job);
    }

    pub fn scope<'a, F, R>(&self, op: F) -> R
    where
        F: FnOnce(&rayon::Scope<'a>) -> R + Send,
        R: Send,
    {
        self.pool.scope(op)
    }

    pub fn install<F, R>(&self, op: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(op)
    }
}
