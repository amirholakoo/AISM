import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import JSONView from '@uiw/react-json-view';

const JsonDisplayBox = ({ data, title = "پاسخ سرور" }) => {
  if (!data) {
    return null;
  }

  return (
    <Card className="bg-white shadow-md">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-slate-900">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
          <JSONView 
            value={data}
            style={{
              backgroundColor: 'transparent',
              fontSize: '14px',
              fontFamily: 'monospace',
              direction: 'ltr',
              textAlign: 'left'
            }}
            displayDataTypes={false}
            displayObjectSize={false}
            enableClipboard={true}
            shortenTextAfterLength={100}
            theme="light"
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default JsonDisplayBox;
