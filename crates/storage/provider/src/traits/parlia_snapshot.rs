use reth_db::models::parlia::Snapshot;
use reth_interfaces::provider::ProviderResult;
use reth_primitives::B256;

/// The trait for fetching parlia snapshot related data.
#[auto_impl::auto_impl(&, Arc)]
pub trait ParliaSnapshotReader: Send + Sync {
    /// Fetch the snapshot for the given block hash.
    fn get_parlia_snapshot(&self, block_hash: B256) -> ProviderResult<Option<Snapshot>>;
}

/// The trait for updating parlia snapshot related data.
#[auto_impl::auto_impl(&, Arc)]
pub trait ParliaSnapshotWriter: Send + Sync {
    /// Save snapshot.
    fn save_parlia_snapshot(&self, block_hash: B256, snapshot: Snapshot) -> ProviderResult<()>;
}
