import React, { useState } from 'react';
import { Button, Input, List } from 'antd';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    if (inputValue.trim() !== '') {
      setMessages([...messages, inputValue]);
      setInputValue('');
    }
  };

  return (
    <div className="chat-container">
      <List
        header={<div>Chat Messages</div>}
        bordered
        dataSource={messages}
        renderItem={item => (
          <List.Item>{item}</List.Item>
        )}
        style={{ width: 300, height: 400, overflow: 'auto' }}
      />
      <Input.Group compact style={{ marginTop: 20 }}>
        <Input
          style={{ width: 'calc(100% - 100px)' }}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onPressEnter={handleSend}
          placeholder="Enter message..."
        />
        <Button type="primary" onClick={handleSend}>Send</Button>
      </Input.Group>
    </div>
  );
}

export default App;
