import * as React from 'react';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';

interface Column {
  id: 'name' | 'retriever' | 'generative' | 'database' | 'maxLength' | 'isSelected'| 'deleteID';
  label: string;
  minWidth?: number;
  align?: 'left';
  format?: (value: number) => string;
}

const columns: readonly Column[] = [
  { id: 'name', label: 'Name', minWidth: 100 },
  { id: 'retriever', label: 'Retrieval Model', minWidth: 100 },
  {
    id: 'generative',
    label: 'Generative Model',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'database',
    label: 'Database Path',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'maxLength',
    label: 'Max Length',
    minWidth: 100,
    align: 'left',
    format: (value: number) => value.toFixed(2),
  },
  {
    id: 'isSelected',
    label: 'isSelected',
    minWidth: 100,
    align: 'left',
  },
  {
    id: 'deleteID',
    label: 'Delete',
    minWidth: 100,
    align: 'left',
  },
];

interface Data {
  name: string;
  retriever: string;
  generative: string;
  database: string;
  maxLength: number;
  isSelected: boolean;
  deleteID: string;
}

function createData(
  name: string,
  retriever: string,
  generative: string,
  database: string,
  maxLength: number,
  isSelected: boolean,
  deleteID: string
): Data {
  return { name, retriever, generative, database, maxLength, isSelected, deleteID };
}

const rows = [
  createData('test name 1', 'test retrieval 1', 'test generative 1', 'test database 1', 2048, false, '1'),
  createData('test name 2', 'test retrieval 2', 'test generative 2', 'test database 2', 2048, true, '2'),
  createData('test name 3', 'test retrieval 3', 'test generative 3', 'test database 3', 2048, false, '3'),
  createData('test name 4', 'test retrieval 4', 'test generative 4', 'test database 4', 2048, false, '4'),
];

export default function StickyHeadTableParam() {
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);

  const handleDelete = (id: string) => {
    console.log(id);
  };

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <TableContainer sx={{ maxHeight: 440 }}>
        <Table stickyHeader aria-label="sticky table" sx={{ background: 'rgba(0, 0, 0, 0.7)', color: 'white' }}>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{
                    minWidth: column.minWidth,
                    color: 'white',
                    background: 'gray',
                    fontWeight: 'bold',
                    fontSize: '1.2rem',
                  }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row) => {
                return (
                  <TableRow hover role="checkbox" tabIndex={-1} key={row.retriever}>
                    {columns.map((column) => {
                      const value = row[column.id];
                      return (
                        <TableCell
                          key={column.id}
                          align={column.align}
                          style={{
                            color: 'white',
                          }}
                        >
                          {column.id === 'isSelected' ? (
                            <input type="radio" checked={Boolean(value)} readOnly />
                          ) : column.id === 'deleteID' ? (
                            <button
                              onClick={() => handleDelete(value.toString())}
                              style={{
                                color: 'white',
                                background: 'red',
                                border: 'none',
                                padding: '5px 10px',
                                borderRadius: '5px',
                                cursor: 'pointer',
                              }}
                            >
                              Delete
                            </button>
                          ) : (
                            column.format && typeof value === 'number' ? column.format(value) : value
                          )}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
}