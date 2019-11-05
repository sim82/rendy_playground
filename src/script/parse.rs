use std::collections::HashMap;
// struct TokenStream<'a, I: Iterator<Item = &'a str>> {
//     iter: std::iter::Peekable<I>,
// }

// impl<'a, I: Iterator<Item = &'a str>> TokenStream<'a, I> {
//     fn new(iter: I) -> Self {
//         Self {
//             iter: iter.peekable(),
//         }
//     }

//     fn peek(&mut self) -> Option<I::Item> {
//         self.iter.peek().and_then(|x| Some(x.clone()))
//     }
// }

//type TokenStream<'a> = Iterator<Item = &'a str>;

// struct TokenStream<'a> {
//     iter: std::iter::Peekable<std::slice::Iter<&'a str> + 'a>,
// }

// type TokenStream = Iterator<Item = &'str>;

#[derive(Clone, Copy)]
enum TokenType {
    Command,
    Variable,
    Value,
}

enum Token {
    Command(String),
    Variable(String),
    Value(String),
}

static COMMANDS: &'static [(&'static str, &'static [TokenType])] = &[
    ("set", &[TokenType::Variable, TokenType::Value] as &[_]),
    ("get", &[TokenType::Variable] as &[_]),
];
struct CommandDispatcher<'a> {
    templates: HashMap<String, (Vec<TokenType>, Box<Fn(&[Token]) + 'a>)>,
}

impl<'a> CommandDispatcher<'a> {
    fn new() -> Self {
        CommandDispatcher {
            templates: HashMap::new(),
        }
    }

    fn add_command<F: Fn(&[Token]) + 'a>(&mut self, command: &str, params: &[TokenType], func: F) {
        self.templates
            .insert(command.into(), (params.into(), Box::new(func)));
    }
}

fn cmd_set(ts: &mut Iterator<Item = &str>) {
    match (ts.next(), ts.next()) {
        (Some(var), Some(value)) => println!("set {} {}", var, value),
        _ => panic!("parse error"),
    }
}

fn cmd_get(ts: &mut Iterator<Item = &str>) {
    match ts.next() {
        Some(var) => println!("get {}", var),
        _ => panic!("parse error"),
    }
}

fn parse(ts: &mut Iterator<Item = &str>) {
    // let ts = ts.peekable();

    match ts.next() {
        Some("set") => cmd_set(ts),
        Some("get") => cmd_get(ts),
        _ => panic!("parse error"),
    }
}

#[cfg(test)]
mod test {
    use super::super::tokenize;
    use super::*;

    fn pass_token_stream(ts: &mut Iterator<Item = &str>) {}

    #[test]
    fn test_token_stream() {
        let tokens = tokenize::tokenize("set bla \"blub blub\"");
        pass_token_stream(&mut tokens.iter().cloned());
        // let mut ts = tokens.iter().peekable();
        parse(&mut tokens.iter().cloned());
        // assert_eq!(ts.peek(), Some(&&"bli"));
    }
}
