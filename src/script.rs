use itertools::Itertools;
use log;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::mpsc::{Receiver, Sender};

pub mod parse;
pub mod tokenize;

pub enum ScriptError {
    ParseError(String),
    UnknownVariable(String),
}

pub type ScriptResult<T> = Result<T, ScriptError>;

#[derive(Default, Clone)]
pub struct Value {
    v: String,
}

#[derive(Clone)]
pub enum BindingAction {
    Update(String, Value, Option<Value>),
}

pub struct Environment {
    subscriptions: Vec<Sender<BindingAction>>,
    pub variables: HashMap<String, Value>,
}

impl Value {
    fn new_internal(v: &str) -> Value {
        Value { v: v.into() }
    }
    pub fn new<T: ToValue>(v: T) -> Value {
        v.to_value()
    }

    pub fn get<T: FromValue>(&self) -> T {
        T::from_value(self)
    }
}

pub trait ToValue {
    fn to_value(&self) -> Value;
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

impl ToValue for i32 {
    fn to_value(&self) -> Value {
        Value {
            v: self.to_string(),
        }
    }
}

impl FromValue for i32 {
    fn from_value(value: &Value) -> Self {
        match Self::from_str(&value.v) {
            Ok(v) => v,
            _ => Self::default(),
        }
    }
}

impl<T: std::fmt::Display + nalgebra::base::Scalar> ToValue for nalgebra::Point3<T> {
    fn to_value(&self) -> Value {
        Value {
            v: format!("({} {} {})", self[0], self[1], self[2]),
        }
    }
}

impl<T: std::str::FromStr + std::default::Default + nalgebra::base::Scalar> FromValue
    for nalgebra::Point3<T>
{
    fn from_value(value: &Value) -> Self {
        let v = &value.v;

        if !(v.starts_with("(") && v.ends_with(")")) {
            return Self::new(T::default(), T::default(), T::default());
        }

        let v = &v[1..v.len() - 1];
        let comps = v.split_whitespace().collect::<Vec<_>>();

        if comps.len() != 3 {
            return Self::new(T::default(), T::default(), T::default());
        }

        match (
            T::from_str(comps[0]),
            T::from_str(comps[1]),
            T::from_str(comps[2]),
        ) {
            (Ok(c0), Ok(c1), Ok(c2)) => Self::new(c0, c1, c2),
            _ => Self::new(T::default(), T::default(), T::default()),
        }
    }
}

impl ToValue for str {
    fn to_value(&self) -> Value {
        Value { v: self.into() }
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> Self {
        match Self::from_str(&value.v) {
            Ok(v) => v.clone(),
            _ => Self::default(),
        }
    }
}

impl<T: ToValue> From<T> for Value {
    fn from(v: T) -> Self {
        v.to_value()
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Display::fmt(&self.v, fmt)
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Debug::fmt(&self.v, fmt)
    }
}

pub struct ValueWatch {
    pub value: Value,
    updated: bool,
}

impl ValueWatch {
    pub fn new() -> Rc<RefCell<ValueWatch>> {
        Rc::new(RefCell::new(ValueWatch {
            value: std::default::Default::default(),
            updated: true,
        }))
    }

    pub fn set(&mut self, v: Value) {
        self.value = v;
        self.updated = true;
    }

    pub fn get_update<T: FromValue>(&mut self) -> Option<T> {
        if self.updated {
            self.updated = false;
            Some(self.value.get::<T>())
        } else {
            None
        }
    }
}

pub struct BindingDispatcher {
    rx: Receiver<BindingAction>,

    callbacks: HashMap<String, Box<FnMut(String, Value, Option<Value>)>>,
    value_bindings: HashMap<String, Rc<RefCell<ValueWatch>>>,
}

impl BindingDispatcher {
    pub fn new(rx: Receiver<BindingAction>) -> Self {
        BindingDispatcher {
            rx: rx,
            callbacks: HashMap::new(),
            // i32_bindings: HashMap::new(),
            value_bindings: HashMap::new(),
        }
    }

    pub fn add_callback<CB: FnMut(String, Value, Option<Value>) + 'static>(
        &mut self,
        name: &str,
        c: CB,
    ) {
        self.callbacks.insert(name.into(), Box::new(c));
    }

    pub fn bind_value(&mut self, name: &str, v: Rc<RefCell<ValueWatch>>) {
        self.value_bindings.insert(name.into(), v);
    }
    pub fn dispatch(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(BindingAction::Update(name, value, old)) => {
                    match self.callbacks.get_mut(&name) {
                        Some(cb) => (*cb)(name.clone(), value.clone(), old),
                        _ => (),
                    }

                    match self.value_bindings.get_mut(&name) {
                        Some(binding) => {
                            // binding.replace(value.parse::<i32>().unwrap());
                            binding.borrow_mut().set(value);
                        }
                        _ => (),
                    }
                }
                _ => break,
            }
        }
    }
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            subscriptions: Vec::new(),
            variables: HashMap::new(),
        }
    }

    pub fn get(&self, str: &str) -> Option<Value> {
        self.variables.get(str).map(|x| x.clone())
    }

    pub fn set(&mut self, name: &str, v: Value) {
        let old = self.variables.insert(name.into(), v.clone());
        self.send_all(BindingAction::Update(name.into(), v.clone(), old));
    }

    pub fn subscribe(&mut self, sender: Sender<BindingAction>) {
        self.subscriptions.push(sender);
    }

    fn send_all(&mut self, action: BindingAction) {
        self.subscriptions
            .drain_filter(|sub| sub.send(action.clone()).is_err());
    }
}

#[derive(Clone)]
pub enum ScriptToken {
    Command(String),
    Variable(String),
    Value(String),
}

pub enum CompletionQuery {
    None,
    Variable(String),
}
// pub struct CompletionDesc(CompletionQuery, usize);

pub fn tokenize(line: &str) -> Option<VecDeque<ScriptToken>> {
    let mut out = VecDeque::new();
    let mut tokens = line.split_whitespace().collect::<VecDeque<_>>();
    if tokens.is_empty() {
        return None;
    }

    let token = tokens.pop_front().unwrap();
    match &token[..] {
        "set" => {
            out.push_back(ScriptToken::Command("set".into()));

            if !tokens.is_empty() {
                out.push_back(ScriptToken::Variable(tokens.pop_front().unwrap().into()));

                if !tokens.is_empty() {
                    out.push_back(ScriptToken::Value(tokens.pop_front().unwrap().into()));
                }
            }
        }
        "get" => {
            out.push_back(ScriptToken::Command("get".into()));

            if !tokens.is_empty() {
                out.push_back(ScriptToken::Variable(tokens.pop_front().unwrap().into()));
            }
        }
        _ => {
            out.push_back(ScriptToken::Command(token.into()));
        }
    }

    Some(out)
}

pub fn parse_internal(line: &str, env: &mut Environment) -> ScriptResult<()> {
    let tokens = tokenize(line);

    if tokens.is_none() {
        return Err(ScriptError::ParseError("not tokens".into()));
    }
    let mut tokens = tokens.unwrap();

    match tokens.pop_front().as_ref() {
        Some(ScriptToken::Command(cmd)) if cmd == "set" => match tokens.pop_front() {
            Some(ScriptToken::Variable(variable)) => match tokens.pop_front() {
                Some(ScriptToken::Value(value)) => {
                    env.set(&variable, value.to_value());
                    Ok(())
                }
                _ => Err(ScriptError::ParseError(format!("expected value"))),
            },
            _ => Err(ScriptError::ParseError("expected varable".into())),
        },
        Some(ScriptToken::Command(cmd)) if cmd == "get" => match tokens.pop_front() {
            Some(ScriptToken::Variable(variable)) => env
                .get(&variable)
                .map_or(Err(ScriptError::UnknownVariable(variable.clone())), |val| {
                    Ok(log::info!("{} = {}", &variable, val))
                }),
            _ => Err(ScriptError::ParseError("expected variable".into())),
        },
        _ => Err(ScriptError::ParseError("expected command".into())),
    }
}

pub fn parse(line: &str, env: &mut Environment) {
    match parse_internal(line, env) {
        Err(ScriptError::ParseError(msg)) => log::warn!("parse error: {}", msg),
        Err(ScriptError::UnknownVariable(var)) => log::warn!("unknown variable: {}", var),
        Ok(()) => (),
    };
}

pub fn complete_generic<'a, I: Iterator<Item = String>>(token: &str, candidates: I) -> Vec<String> {
    let mut completions = Vec::new();

    for key in candidates.map(|x| x.to_string()) {
        if key.len() < token.len() {
            continue;
        }

        if key[..token.len()] == *token {
            completions.push(key.clone());
        }
    }

    completions
}

pub fn produce(tokens: Vec<ScriptToken>) -> String {
    tokens
        .iter()
        .map(|token| match token {
            ScriptToken::Command(cmd) => cmd,
            ScriptToken::Variable(var) => var,
            ScriptToken::Value(val) => val,
        })
        .join(" ")
}

pub fn complete(line: &str, env: &Environment) -> Vec<String> {
    if let Some(mut tokens) = tokenize(line) {
        let completed = match tokens.back() {
            Some(ScriptToken::Command(cmd)) => {
                complete_generic(cmd, vec!["set", "get"].iter().map(|x| x.to_string()))
                    .iter()
                    .map(|x| ScriptToken::Command(x.to_string()))
                    .collect()
            }

            Some(ScriptToken::Variable(var)) => {
                complete_generic(var, env.variables.keys().map(|x| x.to_string()))
                    .iter()
                    .map(|x| ScriptToken::Command(x.to_string()))
                    .collect()
            }
            Some(ScriptToken::Value(val)) => vec![ScriptToken::Value(val.to_string())],
            None => panic!("meeeeep"),
        };

        tokens.pop_back();
        completed
            .iter()
            .map(|comp| {
                let mut tokens = tokens.clone();
                tokens.push_back(comp.clone());

                produce(tokens.into())
            })
            .collect()
    } else {
        vec![line.into()]
    }
}
