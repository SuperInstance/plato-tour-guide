#[cfg(test)]
mod debug_test {
    use crate::{Token, DistanceMatrix, find_maximal_cliques};
    
    #[test]
    fn debug_adjacency() {
        let tokens = vec!["hello", "hallo", "helo", "world"];
        let token_objs: Vec<Token> = tokens.iter().map(|s| Token::new(s.clone())).collect();
        let matrix = DistanceMatrix::new(&token_objs);
        
        let n = matrix.len();
        let threshold = 0.4;
        
        // Build adjacency the same way as the function
        let mut adj = vec![vec![false; n]; n];
        for i in 0..n {
            adj[i][i] = true;
            for j in (i + 1)..n {
                let connected = matrix.get(i, j) <= threshold;
                adj[i][j] = connected;
                adj[j][i] = connected;
            }
        }
        
        eprintln!("DEBUG adjacency matrix (T=0.4):");
        for i in 0..n {
            eprint!("DEBUG   row {}: ", tokens[i]);
            for j in 0..n {
                eprint!("{}", if adj[i][j] { 1 } else { 0 });
            }
            eprintln!();
        }
        
        // Count connected neighbors per node
        for i in 0..n {
            let count = adj[i].iter().filter(|&&v| v).count();
            eprintln!("DEBUG   node {} connections: {}", tokens[i], count);
        }
        
        // Now test with higher threshold
        let cliques_high = find_maximal_cliques(&matrix, 0.7, 2, None);
        eprintln!("DEBUG cliques T=0.7: count={}", cliques_high.count);
        for c in &cliques_high.cliques {
            eprintln!("DEBUG   clique: {:?}", c.members);
        }
    }
}
