#[derive(Clone, Copy, Debug)]
enum State {
    Quoted(usize),
    Identifier(usize),
    Whitespace,
}

pub fn tokenize(line: &str) -> Vec<&str> {
    // let mut iter = line.chars();
    let mut state = State::Whitespace;

    let mut out = Vec::new(); //<&str>::new();

    for (i, c) in line.chars().enumerate() {
        let new_state = match state {
            State::Whitespace => match c {
                '"' => State::Quoted(i),
                c if c.is_whitespace() => State::Whitespace,
                _ => State::Identifier(i),
            },
            State::Identifier(istart) => match c {
                '"' => State::Quoted(i),
                c if c.is_whitespace() => State::Whitespace,
                _ => State::Identifier(istart),
            },
            State::Quoted(istart) => match c {
                '"' => State::Whitespace,
                _ => State::Quoted(istart),
            },
        };

        match (state, new_state) {
            (State::Identifier(istart), State::Whitespace)
            | (State::Identifier(istart), State::Quoted(_)) => out.push(&line[istart..i]),
            (State::Quoted(istart), State::Whitespace) => out.push(&line[istart + 1..i]),
            _ => (),
        }
        state = new_state;
    }

    match state {
        State::Identifier(istart) | State::Quoted(istart) => out.push(&line[istart..]),
        _ => (),
    }

    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(tokenize("bli bla blub"), vec!["bli", "bla", "blub"]);

        assert_eq!(tokenize("bli     bla blub"), vec!["bli", "bla", "blub"]);

        assert_eq!(tokenize("    bli     bla blub"), vec!["bli", "bla", "blub"]);

        assert_eq!(tokenize("bli     bla blub   "), vec!["bli", "bla", "blub"]);

        assert_eq!(
            tokenize("bli \"bla bla\" blub"),
            vec!["bli", "bla bla", "blub"]
        );

        assert_eq!(
            tokenize("bli bla \"bla blub\""),
            vec!["bli", "bla", "bla blub"]
        );

        assert_eq!(
            tokenize("bli bla \"bla blub\"    "),
            vec!["bli", "bla", "bla blub"]
        );

        assert_eq!(
            tokenize("\"bli bla\" bla blub"),
            vec!["bli bla", "bla", "blub"]
        );

        assert_eq!(
            tokenize("   \"bli bla\" bla blub"),
            vec!["bli bla", "bla", "blub"]
        );

        assert_eq!(
            tokenize("bli\"bla\"bla blub"),
            vec!["bli", "bla", "bla", "blub"]
        );
    }

    // #[test]
    // fn test_token_stream() {
    //     let tokens = super::tokenize2("bli bla blub");
    //     let mut ts = super::TokenStream::new(tokens.iter());

    //     assert_eq!(ts.peek(), Some(&&"bli".to_string()));
    // }
}
