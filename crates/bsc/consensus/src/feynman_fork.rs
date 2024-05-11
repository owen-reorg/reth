use reth_primitives::{Address, U256};
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Debug, Eq, PartialEq)]
pub struct ValidatorItem {
    pub address: Address,
    pub voting_power: U256,
    pub vote_address: Vec<u8>,
}

impl Ord for ValidatorItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.voting_power.cmp(&other.voting_power) {
            Ordering::Equal => self.address.cmp(&other.address),
            other => other,
        }
    }
}

impl PartialOrd for ValidatorItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn get_top_validators_by_voting_power(
    consensus_addrs: Vec<Address>,
    voting_powers: Vec<U256>,
    vote_addresses: Vec<Vec<u8>>,
    total_length: U256,
    max_elected_validators: U256,
) -> Option<(Vec<Address>, Vec<U256>, Vec<Vec<u8>>)> {
    if consensus_addrs.len() != total_length.to::<usize>() ||
        voting_powers.len() != total_length.to::<usize>() ||
        vote_addresses.len() != total_length.to::<usize>()
    {
        return None;
    }

    let mut validator_heap: BinaryHeap<ValidatorItem> = BinaryHeap::new();
    for i in 0..consensus_addrs.len() {
        let item = ValidatorItem {
            address: consensus_addrs[i],
            voting_power: voting_powers[i],
            vote_address: vote_addresses[i].clone(),
        };
        if item.voting_power > U256::ZERO {
            validator_heap.push(item);
        }
    }

    let top_n = max_elected_validators.to::<u64>() as usize;
    let top_n = if top_n > validator_heap.len() { validator_heap.len() } else { top_n };
    let mut e_validators = Vec::with_capacity(top_n);
    let mut e_voting_powers = Vec::with_capacity(top_n);
    let mut e_vote_addrs = Vec::with_capacity(top_n);
    for _ in 0..top_n {
        let item = validator_heap.pop().unwrap();
        e_validators.push(item.address);
        // as the decimal in BNB Beacon Chain is 1e8 and in BNB Smart Chain is 1e18, we need to
        // divide it by 1e10
        e_voting_powers.push(item.voting_power / U256::from(10u64.pow(10)));
        e_vote_addrs.push(item.vote_address);
    }

    Some((e_validators, e_voting_powers, e_vote_addrs))
}
