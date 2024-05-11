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
            // If the voting power is the same, we compare the address.
            // Address with smaller value is considered as larger.
            Ordering::Equal => other.address.cmp(&self.address),
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

#[cfg(test)]
mod tests {
    use super::*;
    use reth_primitives::hex;

    #[test]
    fn validator_heap() {
        let test_cases = vec![
            (
                "normal case",
                2,
                vec![
                    ValidatorItem {
                        address: Address::with_last_byte(1),
                        voting_power: U256::from(300) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x01").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(2),
                        voting_power: U256::from(200) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x02").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(3),
                        voting_power: U256::from(100) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x03").unwrap(),
                    },
                ],
                vec![Address::with_last_byte(1), Address::with_last_byte(2)],
            ),
            (
                "same voting power",
                2,
                vec![
                    ValidatorItem {
                        address: Address::with_last_byte(1),
                        voting_power: U256::from(300) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x01").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(2),
                        voting_power: U256::from(100) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x02").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(3),
                        voting_power: U256::from(100) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x03").unwrap(),
                    },
                ],
                vec![Address::with_last_byte(1), Address::with_last_byte(2)],
            ),
            (
                "zero voting power and k > len(validators)",
                5,
                vec![
                    ValidatorItem {
                        address: Address::with_last_byte(1),
                        voting_power: U256::from(300) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x01").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(2),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x02").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(3),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x03").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(4),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x04").unwrap(),
                    },
                ],
                vec![Address::with_last_byte(1)],
            ),
            (
                "zero voting power and k < len(validators)",
                5,
                vec![
                    ValidatorItem {
                        address: Address::with_last_byte(1),
                        voting_power: U256::from(300) * U256::from(10u64.pow(10)),
                        vote_address: hex::decode("0x01").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(2),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x02").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(3),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x03").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(4),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x04").unwrap(),
                    },
                ],
                vec![Address::with_last_byte(1)],
            ),
            (
                "all zero voting power",
                2,
                vec![
                    ValidatorItem {
                        address: Address::with_last_byte(1),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x01").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(2),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x02").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(3),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x03").unwrap(),
                    },
                    ValidatorItem {
                        address: Address::with_last_byte(4),
                        voting_power: U256::ZERO,
                        vote_address: hex::decode("0x04").unwrap(),
                    },
                ],
                vec![],
            ),
        ];

        for (description, k, validators, expected) in test_cases {
            let (eligible_validators, _, _) = get_top_validators_by_voting_power(
                validators.iter().map(|v| v.address).collect(),
                validators.iter().map(|v| v.voting_power).collect(),
                validators.iter().map(|v| v.vote_address.clone()).collect(),
                U256::from(validators.len()),
                U256::from(k),
            )
            .unwrap();

            assert_eq!(eligible_validators.len(), expected.len(), "case: {}", description);
            for i in 0..expected.len() {
                assert_eq!(eligible_validators[i], expected[i], "case: {}", description);
            }
        }
    }
}
